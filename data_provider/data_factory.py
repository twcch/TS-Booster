from functools import partial
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def _loader_options(args, flag):
    device = getattr(args, 'device', None)
    device_type = getattr(device, 'type', str(device).split(':', 1)[0]) if device is not None else getattr(args, 'gpu_type', 'cpu')
    train = flag.lower() == 'train'
    return dict(batch_size=args.batch_size, shuffle=train,
                num_workers=args.num_workers,
                drop_last=train and bool(getattr(args, 'train_drop_last', False)),
                pin_memory=bool(getattr(args, 'pin_memory', False)) and device_type == 'cuda' and bool(getattr(args, 'use_gpu', True)),
                persistent_workers=bool(getattr(args, 'persistent_workers', False)) and args.num_workers > 0)


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    loader_options = _loader_options(args, flag)
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            **loader_options)
        return data_set, data_loader
    elif args.task_name == 'classification':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            **loader_options,
            collate_fn=partial(collate_fn, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            **loader_options)
        return data_set, data_loader
