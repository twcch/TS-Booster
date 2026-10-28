import argparse
import torch
import torch.backends
from utils.print_args import print_args
from copy import deepcopy

def build_parser():
    parser = argparse.ArgumentParser(description='TS-Booster (TSLib-based time-series experiments)')

    parser.add_argument('--rand_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--pred_offset', type=int, default=0,
                        help='skip this many steps after the forecast origin: predict steps '
                             'pred_offset+1 .. pred_offset+pred_len (long_term_forecast, encoder-only models)')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='loss function; long_term_forecast: MSE, MAE, HUBER, HYBRID; '
                             'short_term_forecast: MSE, MAPE, MASE, SMAPE')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='learning-rate policy: type1, type2, type3, cosine, plateau, none')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam',
                        help='long_term_forecast optimizer; other tasks keep their own')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='max gradient norm; unset disables clipping (classification fixes 4.0)')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='lradj=plateau: LR reduction factor')
    parser.add_argument('--lr_patience', type=int, default=5, help='lradj=plateau: epochs without improvement')
    parser.add_argument('--lr_threshold', type=float, default=1e-3,
                        help='lradj=plateau: relative improvement threshold')
    parser.add_argument('--min_lr', type=float, default=0.0, help='lradj=plateau: lower bound of the LR')
    parser.add_argument('--train_drop_last', action='store_true',
                        help='drop the last incomplete training batch (long_term_forecast)')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='loss HUBER/HYBRID: Huber delta')
    parser.add_argument('--dir_weight', type=float, default=0.5, help='loss HYBRID: weight of the direction term')
    parser.add_argument('--dir_threshold', type=float, default=0.01,
                        help='loss HYBRID: minimum |true change| for the direction term')
    parser.add_argument('--train_mse_weight', type=float, default=0.0,
                        help='add w * MSE(forecast, target) to the training loss only; with --loss HYBRID '
                             'and w=0.1 this is TS-Booster\'s backbone-anchoring term for models without experts')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='logical CUDA device ids')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # TS-Booster (models/my_models/tsbooster): TSLib flags cover the rest; see effective_config.json
    parser.add_argument('--tsb_backbone', choices=['linear', 'lstm', 'gru', 'transformer'], default='linear',
                        help='TS-Booster host; LSTM/GRU read d_model/e_layers, Transformer also n_heads/d_ff')
    parser.add_argument('--tsb_decomp_kernels', type=int, nargs='+', default=[7, 13],
                        help='moving-average kernels of the learnable decomposition (odd)')
    parser.add_argument('--tsb_expert_kernels', type=int, nargs='+', default=[3, 5, 7],
                        help='conv kernels of each CNN expert (odd)')
    parser.add_argument('--tsb_expert_hidden', type=int, default=32,
                        help='expert hidden width, split evenly across the kernels')
    parser.add_argument('--tsb_n_experts', type=int, default=2, help='number of boosting experts')
    parser.add_argument('--tsb_expert_type', choices=['cnn', 'mlp'], default='cnn', help='expert architecture')
    parser.add_argument('--tsb_enrich', choices=['trend', 'seasonal', 'both', 'raw', 'zeros'], default='trend',
                        help='expert input: [x || trend], [x || seasonal], [x || trend || seasonal], '
                             'or the width-matched controls [x || x] (raw) and [x || 0] (zeros)')
    parser.add_argument('--tsb_decomp_fixed', action='store_true',
                        help='freeze the decomposition at equal kernel weights (no learnable mixture)')
    parser.add_argument('--tsb_aux_weight', type=float, default=0.3,
                        help='training weight of MSE(y_expert, y - sg(y_base))')
    parser.add_argument('--tsb_base_weight', type=float, default=0.1, help='training weight of MSE(y_base, y)')

    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--anomaly_protocol', choices=['calibration', 'legacy'], default='calibration')
    parser.add_argument('--mask_eval_seed', type=int, default=None)
    parser.add_argument('--metric_scale', choices=['both', 'original', 'normalized'], default='both')
    parser.add_argument('--target_channel_index', type=int, default=-1)
    parser.add_argument('--no_plots', action='store_true')
    parser.add_argument('--diagnostic_test', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--allow_remote_data', action='store_true')
    parser.add_argument('--data_revision', default=None)
    parser.add_argument('--runtime_dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    parser.add_argument('--model_revision', default=None)
    parser.add_argument('--model_cache_dir', default=None)
    parser.add_argument('--local_files_only', action='store_true')
    parser.add_argument('--results_root', default='./results')
    parser.add_argument('--test_results_root', default='./test_results')
    parser.add_argument('--checkpoint_run', default=None, help='saved run directory name for standalone testing')
    parser.add_argument('--batch_key', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--batch_attempt_id', default=None, help=argparse.SUPPRESS)
    return parser


TASKS = {
    'long_term_forecast': ('exp.exp_long_term_forecasting', 'Exp_Long_Term_Forecast'),
    'short_term_forecast': ('exp.exp_short_term_forecasting', 'Exp_Short_Term_Forecast'),
    'imputation': ('exp.exp_imputation', 'Exp_Imputation'),
    'anomaly_detection': ('exp.exp_anomaly_detection', 'Exp_Anomaly_Detection'),
    'classification': ('exp.exp_classification', 'Exp_Classification'),
    'zero_shot_forecast': ('exp.exp_zero_shot_forecasting', 'Exp_Zero_Shot_Forecast'),
}


def validate_args(args):
    if args.task_name not in TASKS:
        raise ValueError(f'未知 task_name={args.task_name!r}；可用任務：{list(TASKS)}')
    if args.is_training not in (0, 1):
        raise ValueError('--is_training 必須為 0 或 1。')
    if args.task_name == 'zero_shot_forecast' and args.is_training:
        raise ValueError('zero_shot_forecast 僅支援 --is_training 0。')
    if args.task_name == 'short_term_forecast' and args.data != 'm4':
        raise ValueError('short_term_forecast 目前支援 M4；一般 CSV 請使用 long_term_forecast。')
    supported_data = {
        'classification': {'UEA'},
        'anomaly_detection': {'PSM', 'MSL', 'SMAP', 'SMD', 'SWAT'},
        'short_term_forecast': {'m4'},
        'long_term_forecast': {'custom', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'},
        'zero_shot_forecast': {'custom', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'},
        'imputation': {'custom', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'},
    }
    if args.data not in supported_data[args.task_name]:
        raise ValueError(f'{args.task_name} 支援的資料類型：{sorted(supported_data[args.task_name])}')
    for key in ('seq_len', 'pred_len', 'batch_size', 'train_epochs', 'itr', 'patience'):
        if getattr(args, key) <= 0:
            raise ValueError(f'--{key} 必須大於零。')
    if not 0 <= args.label_len <= args.seq_len:
        raise ValueError('--label_len 必須介於 0 與 seq_len。')
    if not 0 < args.val_ratio < 1:
        raise ValueError('--val_ratio 必須介於 0 與 1。')
    if args.num_workers < 0:
        raise ValueError('--num_workers 不可為負。')
    if args.persistent_workers and not args.num_workers:
        raise ValueError('--persistent_workers 需要 num_workers > 0。')
    if args.task_name == 'imputation' and not 0 < args.mask_rate < 1:
        raise ValueError('imputation 的 --mask_rate 必須嚴格介於 0 與 1。')
    if args.task_name == 'imputation' and args.seq_len < 2:
        raise ValueError('imputation 需要 seq_len >= 2。')
    if args.task_name == 'anomaly_detection' and not 0 < args.anomaly_ratio < 100:
        raise ValueError('--anomaly_ratio 必須介於 0 與 100（百分比）。')
    if args.allow_remote_data and not args.data_revision:
        raise ValueError('--allow_remote_data 需要固定 --data_revision。')
    if args.checkpoint_run and args.is_training:
        raise ValueError('--checkpoint_run 僅供獨立測試使用。')
    if args.checkpoint_run and args.task_name == 'zero_shot_forecast':
        raise ValueError('--checkpoint_run 僅供已訓練的監督式模型使用；foundation 請指定 model_revision。')
    if args.features == 'MS' and args.task_name in {'long_term_forecast', 'zero_shot_forecast'}:
        if args.target_channel_index not in (-1, args.enc_in - 1):
            raise ValueError('CSV 的 MS 資料會將 target 排到最後；target_channel_index 必須為 -1 或 enc_in-1。')
    if args.learning_rate <= 0:
        raise ValueError('--learning_rate 必須大於零。')
    validate_optimization_args(args)
    from models.capabilities import validate_model_task
    capability = validate_model_task(args.model, args.task_name)
    if capability.foundation and args.use_multi_gpu:
        raise ValueError('Foundation provider 目前支援單裝置；不支援 --use_multi_gpu。')


def validate_optimization_args(args):
    from exp.common.training import LR_POLICIES
    from utils.forecast_losses import FORECAST_LOSSES
    if args.lradj not in LR_POLICIES:
        raise ValueError(f'--lradj 必須為 {LR_POLICIES} 之一。')
    if args.task_name == 'long_term_forecast' and args.loss.upper() not in FORECAST_LOSSES:
        raise ValueError(f'long_term_forecast 的 --loss 必須為 {FORECAST_LOSSES} 之一。')
    if args.task_name != 'long_term_forecast' and (
            args.optimizer != 'adam' or args.weight_decay or args.train_drop_last):
        raise ValueError('--optimizer/--weight_decay/--train_drop_last 目前只作用於 long_term_forecast。')
    if args.weight_decay < 0:
        raise ValueError('--weight_decay 不可為負。')
    if args.clip_grad is not None and args.clip_grad <= 0:
        raise ValueError('--clip_grad 必須大於零；不設定即關閉。')
    if not 0 < args.lr_factor < 1 or args.lr_patience < 0 or args.lr_threshold < 0 or args.min_lr < 0:
        raise ValueError('plateau 參數需 0 < lr_factor < 1，且 lr_patience、lr_threshold、min_lr 不可為負。')
    if args.huber_delta <= 0 or not 0 <= args.dir_weight <= 1 or args.dir_threshold < 0:
        raise ValueError('需 huber_delta > 0、0 <= dir_weight <= 1、dir_threshold >= 0。')
    kernels = list(args.tsb_decomp_kernels) + list(args.tsb_expert_kernels)
    if any(k <= 0 or k % 2 == 0 for k in kernels):
        raise ValueError('--tsb_decomp_kernels 與 --tsb_expert_kernels 必須為正奇數。')
    if args.tsb_n_experts < 1 or args.tsb_expert_hidden < len(args.tsb_expert_kernels):
        raise ValueError('--tsb_n_experts 至少為 1，--tsb_expert_hidden 不可小於 expert kernel 數。')
    if args.tsb_aux_weight < 0 or args.tsb_base_weight < 0:
        raise ValueError('--tsb_aux_weight 與 --tsb_base_weight 不可為負。')
    if args.train_mse_weight < 0:
        raise ValueError('--train_mse_weight 不可為負。')
    if args.train_mse_weight and args.task_name != 'long_term_forecast':
        raise ValueError('--train_mse_weight 目前只作用於 long_term_forecast。')
    if args.pred_offset < 0:
        raise ValueError('--pred_offset 不可為負。')
    if args.pred_offset:
        if args.task_name != 'long_term_forecast':
            raise ValueError('--pred_offset 目前只支援 long_term_forecast。')
        from models.capabilities import get_model_capability
        capability = get_model_capability(args.model)
        if capability.requires_decoder or capability.requires_time_marks:
            raise ValueError(f'{args.model} 需要 decoder 輸入或時間標記，不支援 --pred_offset。')


def legacy_setting(args, iteration=0):
    offset = getattr(args, 'pred_offset', 0)
    horizon = f'{args.pred_len}_po{offset}' if offset else args.pred_len
    return ('{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_seed{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, horizon, args.d_model, args.n_heads,
        args.e_layers, args.d_layers, args.d_ff, args.expand, args.d_conv,
        args.factor, args.embed, args.distil, args.des, args.rand_seed, iteration))


def main(argv=None):
    import importlib
    from pathlib import Path
    from exp.common.runtime import (TrainResult, resolve_device, seed_everything,
                                    prepare_run, mark_run_failed, precision_context)
    parser = build_parser()
    requested_args = parser.parse_args(argv)
    requested_config = deepcopy(vars(requested_args))
    args = deepcopy(requested_args)
    try:
        validate_args(args)
        resolve_device(args)
        # Validate precision before model construction or downloads.
        with precision_context(args, args.device):
            pass
        if args.checkpoint_run and args.task_name != 'zero_shot_forecast':
            if not (Path(args.checkpoints) / args.checkpoint_run / 'checkpoint.pth').is_file():
                raise ValueError('指定的 checkpoint_run 找不到 checkpoint.pth。')
    except ValueError as error:
        parser.error(str(error))
    module_name, class_name = TASKS[args.task_name]
    experiment_class = getattr(importlib.import_module(module_name), class_name)
    base_args = args
    for iteration in range(base_args.itr if base_args.is_training else 1):
        args = deepcopy(base_args)
        args.requested_config = requested_config
        args.rand_seed = base_args.rand_seed + iteration
        seed_everything(args.rand_seed, args.deterministic)
        exp = experiment_class(args)
        name = legacy_setting(args, iteration)
        if not args.is_training and args.task_name != 'zero_shot_forecast' and not args.checkpoint_run:
            if (Path(args.checkpoints) / name / 'checkpoint.pth').is_file():
                args.checkpoint_run = name
            else:
                parser.error('獨立測試需要 --checkpoint_run 指定已訓練的目錄名稱。')
        print('Resolved args in experiment:')
        print_args(args)
        setting = prepare_run(args, exp.model, name, iteration)
        try:
            if args.is_training:
                trained = exp.train(setting)
                if not isinstance(trained, TrainResult):
                    raise TypeError(f'{args.task_name}.train 必須回傳 TrainResult。')
                exp.test(setting, train_time=trained.train_time_s)
            else:
                exp.test(setting, test=1)
        except BaseException as error:
            mark_run_failed(args, error)
            raise
        if args.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif args.device.type == 'mps':
            torch.mps.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
