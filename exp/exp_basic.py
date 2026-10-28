import os
import torch
import importlib
import torch.nn as nn
from exp.common.runtime import resolve_device

# Just put your model files under models/ folder
# e.g., models/Transformer.py, models/LSTM.py, etc.
# All models will be automatically detected and can be used by specifying their names.

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args

        # -------------------------------------------------------
        #  Automatically generate model map
        # -------------------------------------------------------
        model_map = self._scan_models_directory()

        # Use smart dictionary
        self.model_dict = LazyModelDict(model_map)

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _scan_models_directory(self):
        """遞迴掃描 models/ 資料夾及所有子資料夾，將檔案名稱對應到模組路徑"""
        model_map = {}
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

        for root, dirs, files in os.walk(models_dir):
            # 跳過 __pycache__，並排序以確保順序一致
            dirs[:] = sorted([d for d in dirs if d != '__pycache__'])
            files = sorted(files)

            for filename in files:
                if not filename.endswith('.py') or filename.startswith('_') or filename == 'capabilities.py':
                    continue

                # 用檔案名稱（不含 .py）當作模型名稱
                model_name = filename[:-3]

                # 計算相對於專案根目錄的模組路徑
                # 例如 models/tslib_models/PatchTST.py -> models.tslib_models.PatchTST
                # 例如 models/my_models/tsbooster/TSBooster.py -> models.my_models.tsbooster.TSBooster
                rel_path = os.path.relpath(os.path.join(root, filename), os.path.dirname(models_dir))
                module_path = rel_path.replace(os.sep, '.')[:-3]  # 去掉 .py

                if model_name in model_map:
                    raise ValueError(f'模型名稱 {model_name!r} 重複：{module_path} / {model_map[model_name]}')

                model_map[model_name] = module_path

        return model_map

    def _build_model(self):
        if self.args.model not in self.model_dict.model_map:
            raise ValueError(
                f"模型 '{self.args.model}' 找不到。可用模型: {list(self.model_dict.model_map.keys())}"
            )
        from models.capabilities import get_model_capability
        capability = get_model_capability(self.args.model)
        if capability.foundation and self.args.use_multi_gpu:
            raise ValueError('Foundation providers require a single device.')
        model = self.model_dict[self.args.model](self.args)
        if not capability.foundation:
            model = model.float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        device = resolve_device(self.args)
        print(f'Use device: {device}')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class LazyModelDict(dict):
    """
    Smart Lazy-Loading Dictionary
    """
    def __init__(self, model_map):
        self.model_map = model_map
        super().__init__()

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)

        if key not in self.model_map:
            raise ValueError(f"Model [{key}] not found; available: {sorted(self.model_map)}")

        module_path = self.model_map[key]
        try:
            print(f"🚀 Lazy Loading: {key} ...")
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"❌ Error: Failed to import model [{key}]. Dependencies missing?")
            raise e

        # Try to find the model class
        if hasattr(module, 'Model'):
            model_class = module.Model
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(f"Module {module_path} has no class 'Model' or '{key}'")

        self[key] = model_class
        return model_class
