"""Small training primitives shared by task-specific experiments."""
import math
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from exp.common.runtime import (
    TrainResult, atomic_save_checkpoint, file_fingerprint, load_checkpoint, precision_context, synchronize,
)
from utils.tools import adjust_learning_rate

LR_POLICIES = ("type1", "type2", "type3", "cosine", "plateau", "none")


def build_optimizer(model, args):
    """--optimizer adam (TSLib default) or adamw, with optional --weight_decay."""
    name = getattr(args, "optimizer", "adam")
    weight_decay = getattr(args, "weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer {name!r}; choose adam or adamw")


def build_plateau_scheduler(optimizer, args):
    """ReduceLROnPlateau on the checkpoint-selection metric (lower is better)."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=getattr(args, "lr_factor", 0.5), patience=getattr(args, "lr_patience", 5),
        threshold=getattr(args, "lr_threshold", 1e-3), min_lr=getattr(args, "min_lr", 0.0))


class WeightedLoss:
    """Aggregate batch means by the number of effective loss elements."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, loss, count):
        value = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        if count <= 0 or not math.isfinite(value):
            raise ValueError('Loss requires a positive effective count and a finite value.')
        self.total += value * count
        self.count += count

    def mean(self):
        if not self.count:
            raise ValueError('No effective samples were available for loss evaluation.')
        return self.total / self.count


def require_same_shape(prediction, target):
    if not torch.is_tensor(prediction):
        raise TypeError('The selected model must return a prediction tensor for this task.')
    if prediction.shape != target.shape:
        raise ValueError(f'Prediction shape {tuple(prediction.shape)} does not match '
                         f'target shape {tuple(target.shape)}; broadcasting is not supported.')


def _save_training_state(experiment, setting, optimizer, scaler, epoch, best_value, adjust_lr, plateau=None):
    """Save a weights-only-loadable epoch snapshot; automatic resume is not implemented."""
    numpy_state = np.random.get_state()
    rng = {'python': random.getstate(), 'torch_cpu': torch.get_rng_state(),
           'numpy': {'algorithm': numpy_state[0],
                     'keys': torch.as_tensor(numpy_state[1].astype(np.int64)),
                     'position': int(numpy_state[2]), 'has_gauss': int(numpy_state[3]),
                     'cached_gaussian': float(numpy_state[4])}}
    if experiment.device.type == 'cuda':
        rng['torch_cuda'] = torch.cuda.get_rng_state_all()
    if experiment.device.type == 'mps' and hasattr(torch.mps, 'get_rng_state'):
        rng['torch_mps'] = torch.mps.get_rng_state()
    directory = Path(experiment.args.checkpoints) / setting
    state = {'schema_version': 1, 'model_checkpoint': 'checkpoint.pth',
             'model_checkpoint_sha256': file_fingerprint(directory / 'checkpoint.pth'),
             'epoch': epoch, 'best_metric': best_value,
             'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
             'scheduler': {'enabled': bool(adjust_lr),
                           'policy': getattr(experiment.args, 'lradj', None) if adjust_lr else None,
                           'next_adjustment_epoch': epoch if adjust_lr else None,
                           'plateau_state': plateau.state_dict() if plateau is not None else None},
             'precision': {'autocast_enabled': bool(experiment.args.use_amp),
                           'autocast_dtype': (str(torch.get_autocast_dtype(experiment.device.type))
                                              if experiment.args.use_amp else None),
                           'grad_scaler_enabled': scaler.is_enabled()},
             'rng': rng, 'phase': ('after validation, before learning-rate adjustment'
                                  if adjust_lr else 'after validation; scheduler disabled'),
             'resume_supported': False}
    handle, temporary = tempfile.mkstemp(prefix='.training_state.', dir=directory)
    os.close(handle)
    try:
        torch.save(state, temporary)
        os.replace(temporary, directory / 'training_state.pth')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def train_model(experiment, setting, loader, batch_loss, validate, *,
                select_metric=None, diagnostic=None, clip_grad=None, adjust_lr=True):
    """Share optimizer/checkpoint mechanics, leaving losses and selection to each task.

    ``batch_loss(batch)`` returns (mean loss, effective element count), and
    ``validate()`` may return either a loss or a task-specific validation tuple.
    Only the explicit diagnostic callback is allowed to inspect the test split.
    """
    args, model, device = experiment.args, experiment.model, experiment.device
    if args.train_epochs < 1 or args.patience < 1:
        raise ValueError('train_epochs and patience must be positive.')
    if clip_grad is None:
        clip_grad = getattr(args, 'clip_grad', None)
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    optimizer = experiment._select_optimizer()
    policy = getattr(args, 'lradj', 'type1') if adjust_lr else None
    plateau = build_plateau_scheduler(optimizer, args) if policy == 'plateau' else None
    use_scaler = bool(args.use_amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    best_value, best_epoch, stale_epochs = None, None, 0
    history = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(args.train_epochs):
        model.train()
        train_loss = WeightedLoss()
        synchronize(device)
        epoch_start = time.perf_counter()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with precision_context(args, device):
                loss, count = batch_loss(batch)
            train_loss.update(loss, count)
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        synchronize(device)
        train_s = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        validation = validate()
        synchronize(device)
        validation_s = time.perf_counter() - validation_start
        value = float(select_metric(validation) if select_metric else validation)
        if not math.isfinite(value):
            raise ValueError('The checkpoint selection metric must be finite.')
        record = {'epoch': epoch + 1, 'train_loss': train_loss.mean(),
                  'validation': validation, 'train_s': train_s,
                  'validation_s': validation_s, 'lr': optimizer.param_groups[0]['lr']}
        if diagnostic is not None:
            record['diagnostic_test'] = diagnostic()
        checkpoint_start = time.perf_counter()
        # Preserve the existing EarlyStopping treatment of equal scores.
        if best_value is None or value <= best_value:
            best_value, best_epoch, stale_epochs = value, epoch + 1, 0
            atomic_save_checkpoint(model, args, setting)
            _save_training_state(experiment, setting, optimizer, scaler, best_epoch, best_value, adjust_lr,
                                 plateau)
        else:
            stale_epochs += 1
        record['checkpoint_s'] = time.perf_counter() - checkpoint_start
        history.append(record)
        print(f'Epoch {epoch + 1}: train_loss={record["train_loss"]:.7f}, '
              f'validation={validation}, lr={record["lr"]:.3g}, train_s={train_s:.3f}, val_s={validation_s:.3f}')
        if stale_epochs >= args.patience:
            print('Early stopping')
            break
        if plateau is not None:
            plateau.step(value)
        elif policy not in (None, 'none'):
            adjust_learning_rate(optimizer, epoch + 1, args)
    current_args = SimpleNamespace(**vars(args))
    current_args.checkpoint_run = None
    load_checkpoint(model, current_args, setting, device)
    synchronize(device)
    elapsed = time.perf_counter() - start
    experiment.training_history = history
    experiment.best_epoch = best_epoch
    experiment.best_selection_metric = best_value
    experiment.train_peak_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 ** 2
        if device.type == 'cuda' else None
    )
    return TrainResult(model, elapsed, best_epoch)
