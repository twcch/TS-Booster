"""Forecast evaluation and artifact helpers with explicit metric spaces."""
import os
import sys
import time

import numpy as np
import torch

from exp.common.runtime import (atomic_save_array, load_checkpoint, precision_context, save_evaluation,
                                unwrap_model)
from exp.common.training import require_same_shape, synchronize
from utils.metrics import MDA, metric, metric_report
from utils.tools import visual

METRIC_NAMES = ('mae', 'mse', 'rmse', 'mape', 'mspe', 'r2')
SUMMARY_METRICS = ('mse', 'rmse', 'mae', 'mape', 'mspe', 'r2', 'mda')


def regression_metrics(predictions, targets, anchors=None):
    metrics = dict(zip(METRIC_NAMES, map(float, metric(predictions, targets))))
    if anchors is not None:
        metrics['mda'] = MDA(predictions, targets, anchors)
    return metrics


def plot_directory(args, setting):
    path = os.path.join(getattr(args, 'test_results_root', './test_results'), setting)
    os.makedirs(path, exist_ok=True)
    return path


def forecast_batch(experiment, batch):
    x, y, x_mark, y_mark = [value.float().to(experiment.device) for value in batch]
    args = experiment.args
    decoder = torch.cat((y[:, :args.label_len],
                         torch.zeros_like(y[:, -args.pred_len:])), dim=1)
    return x, y, x_mark, y_mark, decoder


def _first_target_channel(args):
    return -1 if args.features == 'MS' else 0


def forecast_targets(args, outputs, targets):
    if not torch.is_tensor(outputs):
        raise TypeError('The selected model does not return a forecast tensor.')
    first = _first_target_channel(args)
    outputs = outputs[:, -args.pred_len:, first:]
    targets = targets[:, -args.pred_len:, first:]
    require_same_shape(outputs, targets)
    return outputs, targets


def last_observed(args, x):
    """Last input value of every forecast channel, [batch, 1, channel], to broadcast over the horizon."""
    return x[:, -1:, _first_target_channel(args):]


def metric_spaces(args, dataset, predictions, targets, mask=None, anchors=None):
    """Return primary arrays/metrics and explicitly named alternative spaces.

    ``anchors`` ([sample, channel], the last observed values) add the directional
    accuracy (mda); the primary-space anchors are returned in extra['_anchors'].
    """
    requested = getattr(args, 'metric_scale', 'both')
    if requested not in {'both', 'original', 'normalized'}:
        raise ValueError('metric_scale must be both, original, or normalized.')
    is_scaled = bool(getattr(dataset, 'scale', False))
    use_original = bool(getattr(args, 'inverse', False) or requested == 'original')
    def score(prediction, target, anchor):
        if mask is not None:
            prediction, target = prediction[mask == 0], target[mask == 0]
        return regression_metrics(prediction, target, anchor)

    def report(prediction, target, anchor):
        if mask is not None:
            prediction, target = prediction[mask == 0], target[mask == 0]
        return metric_report(prediction, target, anchor)

    normalized = score(predictions, targets, anchors)
    extra = {'metric_space': 'original' if use_original or not is_scaled else 'normalized',
             'metric_scale_requested': requested}
    if requested != 'original' or not is_scaled:
        extra['metrics_normalized' if is_scaled else 'metrics_original'] = normalized
        extra['metric_report_normalized' if is_scaled else 'metric_report_original'] = report(
            predictions, targets, anchors)
    original_predictions, original_targets, original_anchors = predictions, targets, anchors
    if is_scaled and (requested != 'normalized' or use_original):
        target_only = args.features == 'MS'
        original_predictions = dataset.inverse_transform(predictions, target_only=target_only)
        original_targets = dataset.inverse_transform(targets, target_only=target_only)
        if anchors is not None:
            original_anchors = dataset.inverse_transform(anchors, target_only=target_only)
        extra['metrics_original'] = score(original_predictions, original_targets, original_anchors)
        extra['metric_report_original'] = report(original_predictions, original_targets, original_anchors)
    if use_original and is_scaled:
        if anchors is not None:
            extra['_anchors'] = original_anchors
        return original_predictions, original_targets, extra['metrics_original'], extra
    if anchors is not None:
        extra['_anchors'] = anchors
    return predictions, targets, normalized, extra


def runtime_parameter_stats(model):
    """Include provider modules hidden in non-Module pipelines, without duplicates."""
    parameters, buffers = {}, {}
    seen = set()
    holder = model
    # Current provider adapters use wrapper.model or wrapper.model.model.
    # Do not inspect arbitrary properties or recursively walk external APIs.
    for _ in range(3):
        if holder is None or id(holder) in seen:
            break
        seen.add(id(holder))
        if isinstance(holder, torch.nn.Module):
            parameters.update({id(value): value for value in holder.parameters()})
            buffers.update({id(value): value for value in holder.buffers()})
        holder = getattr(holder, 'model', None)
    tensors = list(parameters.values()) + list(buffers.values())
    return {
        'params': sum(p.numel() for p in parameters.values()),
        'trainable_params': sum(p.numel() for p in parameters.values() if p.requires_grad),
        'model_size_mb': sum(t.numel() * t.element_size() for t in tensors) / 1024 ** 2,
        'floating_parameter_dtypes': sorted({str(p.dtype) for p in parameters.values() if p.is_floating_point()}),
        'floating_buffer_dtypes': sorted({str(p.dtype) for p in buffers.values() if p.is_floating_point()}),
        'parameter_count_scope': 'registered modules and provider model modules, deduplicated',
    }


def process_peak_rss_mb():
    """Peak resident memory of this process (CPU side); None where unavailable."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return peak / 1024 ** 2 if sys.platform == 'darwin' else peak / 1024


def evaluation_stats(experiment, train_time, inference_s, sample_count, elapsed_s):
    model, device = experiment.model, experiment.device
    history = getattr(experiment, 'training_history', None)
    stats = {
        'train_time_s': train_time,
        'training_history': history,
        'epochs_trained': len(history) if history else None,
        'best_epoch': getattr(experiment, 'best_epoch', None),
        'best_selection_metric': getattr(experiment, 'best_selection_metric', None),
        'train_timing_scope': 'epoch train/validation/optional diagnostic/checkpoint/LR adjustment and best-weight reload; excludes model/data preparation and optimizer creation',
        'evaluation_compute_s': elapsed_s,
        'evaluation_scope': 'loader iteration, transfers, forward, inverse, metrics, DTW, sampled plots; excludes dataset/loader preparation, artifact save and publication figures',
        'forward_time_s': inference_s,
        'forward_ms_per_sample': inference_s * 1000 / sample_count,
        'forward_samples_per_s': sample_count / inference_s if inference_s > 0 else None,
        'num_samples': sample_count,
        'forward_timing': 'all batches including first; backend synchronized; excludes transfers',
        'autocast_enabled': bool(getattr(experiment.args, 'use_amp', False)),
        'autocast_dtype': (str(torch.get_autocast_dtype(device.type))
                           if getattr(experiment.args, 'use_amp', False) else None),
        'train_peak_allocated_mb': getattr(experiment, 'train_peak_allocated_mb', None),
        'eval_peak_allocated_mb': (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                                   if device.type == 'cuda' else None),
        'process_peak_rss_mb': process_peak_rss_mb(),
    }
    stats.update(runtime_parameter_stats(model))
    return stats


def run_summary(metrics, extra):
    """Flat headline numbers for result.json; full detail stays in details."""
    summary = {'metric_space': extra.get('metric_space')}
    summary.update({name: metrics.get(name) for name in SUMMARY_METRICS if name in metrics})
    for space in ('original', 'normalized'):
        values = extra.get(f'metrics_{space}')
        if values is not None:
            summary[f'metrics_{space}'] = {name: values.get(name) for name in SUMMARY_METRICS if name in values}
    summary.update({
        'train_time_s': extra.get('train_time_s'),
        'inference_time_s': extra.get('forward_time_s'),
        'inference_ms_per_sample': extra.get('forward_ms_per_sample'),
        'inference_samples_per_s': extra.get('forward_samples_per_s'),
        'evaluation_time_s': extra.get('evaluation_compute_s'),
        'params': extra.get('params'), 'trainable_params': extra.get('trainable_params'),
        'model_size_mb': extra.get('model_size_mb'),
        'epochs_trained': extra.get('epochs_trained'), 'best_epoch': extra.get('best_epoch'),
        'train_peak_gpu_mb': extra.get('train_peak_allocated_mb'),
        'eval_peak_gpu_mb': extra.get('eval_peak_allocated_mb'),
        'process_peak_rss_mb': extra.get('process_peak_rss_mb'),
    })
    return summary


class ForecastEvaluationMixin:
    """Long-term and pretrained forecast evaluation share the same data contract."""

    def test(self, setting, test=0, train_time=None):
        test_data, loader = self._get_data(flag='test')
        if test and self.args.task_name != 'zero_shot_forecast':
            load_checkpoint(self.model, self.args, setting, self.device)
        model = unwrap_model(self.model)
        # Models such as TS-Booster report per-sample internals (the expert gate) on request.
        with_aux = bool(getattr(model, 'supports_aux_outputs', False))
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        synchronize(self.device)
        started = time.perf_counter()
        predictions, targets, anchors, gates, plot_inputs = [], [], [], [], []
        inference_s, sample_count = 0.0, 0
        self.model.eval()
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                x, y, x_mark, y_mark, decoder = forecast_batch(self, batch)
                synchronize(self.device)
                before = time.perf_counter()
                with precision_context(self.args, self.device):
                    if with_aux:
                        output, aux = self.model(x, x_mark, decoder, y_mark, return_aux=True)
                    else:
                        output = self.model(x, x_mark, decoder, y_mark)
                synchronize(self.device)
                inference_s += time.perf_counter() - before
                sample_count += x.size(0)
                prediction, target = forecast_targets(self.args, output, y)
                predictions.append(prediction.float().cpu().numpy())
                targets.append(target.float().cpu().numpy())
                anchors.append(last_observed(self.args, x)[:, 0].float().cpu().numpy())
                if with_aux and aux.get('gate') is not None:
                    gates.append(aux['gate'].float().cpu().numpy())
                if not getattr(self.args, 'no_plots', False) and index % 20 == 0:
                    plot_inputs.append((sample_count - x.size(0), index,
                                        x[0].float().cpu().numpy()))
        if not predictions:
            raise ValueError('No test samples are available.')
        predictions, targets = np.concatenate(predictions), np.concatenate(targets)
        predictions, targets, metrics, extra = metric_spaces(
            self.args, test_data, predictions, targets, anchors=np.concatenate(anchors))
        primary_anchors = extra.pop('_anchors', None)
        if getattr(self.args, 'use_dtw', False):
            from utils.dtw_metric import accelerated_dtw
            metrics['dtw'] = float(np.mean([
                accelerated_dtw(p.reshape(-1, 1), t.reshape(-1, 1), dist='cityblock')[0]
                for p, t in zip(predictions, targets)
            ]))
        if not getattr(self.args, 'no_plots', False):
            directory = plot_directory(self.args, setting)
            for sample, index, inputs in plot_inputs:
                if extra['metric_space'] == 'original' and getattr(test_data, 'scale', False):
                    inputs = test_data.inverse_transform(inputs)
                visual(np.concatenate((inputs[:, -1], targets[sample, :, -1])),
                       np.concatenate((inputs[:, -1], predictions[sample, :, -1])),
                       os.path.join(directory, f'{index}.pdf'))
        synchronize(self.device)
        extra.update(evaluation_stats(self, train_time, inference_s, sample_count,
                                      time.perf_counter() - started))
        results = os.path.join(getattr(self.args, 'results_root', './results'), setting)
        atomic_save_array(os.path.join(results, 'last.npy'), np.asarray(primary_anchors))
        extra['additional_artifacts'] = ['last.npy']
        extra['last_observed'] = 'last.npy: last input value of each forecast channel, primary metric space'
        diagnostics = model.diagnostics() if callable(getattr(model, 'diagnostics', None)) else None
        if gates:
            gate = np.concatenate(gates)
            atomic_save_array(os.path.join(results, 'gate.npy'), gate)
            extra['additional_artifacts'].append('gate.npy')
            diagnostics = dict(diagnostics or {})
            diagnostics.update(gate_mean=float(gate.mean()), gate_std=float(gate.std()),
                               gate_min=float(gate.min()), gate_max=float(gate.max()))
        if diagnostics is not None:
            extra['model_diagnostics'] = diagnostics
        result = save_evaluation(self.args, setting, metrics, predictions, targets, extra=extra,
                                 summary=run_summary(metrics, extra))
        if not getattr(self.args, 'no_plots', False):
            try:
                from utils.visualization import generate_figures
                generate_figures(input_path=results, output_path=plot_directory(self.args, setting))
            except (ImportError, OSError, ValueError) as error:
                print(f'Warning: figure generation failed: {error}')
        print(', '.join(f'{name}:{value}' for name, value in metrics.items()))
        return result
