"""Provider runtime options, without importing or downloading provider weights."""

from dataclasses import dataclass
from importlib import import_module
import inspect

import torch


@dataclass(frozen=True)
class FoundationRuntime:
    device: torch.device
    dtype: torch.dtype
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False

    @classmethod
    def from_config(cls, configs, *, fp32_only=False):
        device = torch.device(getattr(configs, "runtime_device", None)
                              or getattr(configs, "device", "cpu"))
        value = getattr(configs, "runtime_dtype", None) or "float32"
        if value == "auto":
            value = "float32"
        dtypes = {"float32": torch.float32, "float16": torch.float16,
                  "bfloat16": torch.bfloat16, "fp32": torch.float32,
                  "fp16": torch.float16, "bf16": torch.bfloat16}
        dtype = value if isinstance(value, torch.dtype) else dtypes.get(value)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported foundation dtype: {value!r}")
        if fp32_only and dtype != torch.float32:
            raise ValueError("This provider adapter supports runtime_dtype=float32 only")
        return cls(device, dtype, getattr(configs, "model_revision", None),
                   getattr(configs, "model_cache_dir", None),
                   bool(getattr(configs, "local_files_only", False)))

    def hub_kwargs(self):
        kwargs = {"local_files_only": self.local_files_only}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        return kwargs

    def metadata(self, model_id):
        return {"model_id": model_id, "model_revision": self.revision,
                "revision_policy": "explicit_reference" if self.revision else "unpinned_default",
                "runtime_device": str(self.device), "runtime_dtype": str(self.dtype),
                "model_cache_dir": self.cache_dir,
                "local_files_only": self.local_files_only}


def require_provider(name):
    try:
        return import_module(name)
    except ImportError as exc:
        raise ImportError(f"Foundation provider {name!r} is unavailable; install its optional dependency group") from exc


def require_zero_shot(configs):
    if configs.task_name != "zero_shot_forecast":
        raise ValueError(f"Foundation wrapper supports zero_shot_forecast, got {configs.task_name!r}")


def normalize_context(x):
    """Preserve the wrappers' existing external standardization formula."""
    mean = x.mean(1, keepdim=True).detach()
    centered = x - mean
    stdev = torch.sqrt(centered.var(1, keepdim=True, unbiased=False) + 1e-5)
    return centered / stdev, mean, stdev


def timesfm_compatible_class(provider_class):
    """Local subclass for old TimesFM constructors; never mutate the provider."""
    signature = inspect.signature(provider_class.__init__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return provider_class
    allowed = {"proxies", "resume_download", "force_download", "local_files_only",
               "token", "revision", "cache_dir"}

    class HubCompatibleTimesFM(provider_class):
        def __init__(self, torch_compile=True, config=None, **hub_kwargs):
            unknown = set(hub_kwargs) - allowed
            if unknown:
                raise TypeError(f"Unsupported TimesFM constructor arguments: {sorted(unknown)}")
            super().__init__(torch_compile=torch_compile, config=config)

    return HubCompatibleTimesFM
