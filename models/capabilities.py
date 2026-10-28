"""Explicit model capabilities, derived from current forward dispatch.

Capability describes implemented tasks, not a claim that every external provider
or benchmark has been integration-tested in the current environment.
"""

from dataclasses import dataclass
from pathlib import Path
import ast


MODEL_ROOT = Path(__file__).resolve().parent
_VALID_TASKS = {"long_term_forecast", "short_term_forecast", "zero_shot_forecast",
                "imputation", "anomaly_detection", "classification"}


@dataclass(frozen=True)
class ModelCapability:
    supported_tasks: tuple[str, ...]
    requires_decoder: bool = False
    requires_time_marks: bool = False
    output_feature_mode: str = "model_defined"
    optional_dependencies: tuple[str, ...] = ()
    foundation: bool = False
    evidence: str = "current forward dispatch"


_TASK_GROUPS = {
    ('anomaly_detection',): ('KANAD',),
    ('anomaly_detection', 'classification', 'imputation', 'long_term_forecast', 'short_term_forecast'): ('Autoformer', 'Crossformer', 'DLinear', 'ETSformer', 'FEDformer', 'FiLM', 'Informer', 'LightTS', 'MICN', 'MSGNet', 'Nonstationary_Transformer', 'PatchTST', 'Pyraformer', 'Reformer', 'SegRNN', 'TimeFilter', 'TimeMixer', 'TimesNet', 'Transformer', 'iTransformer'),
    ('classification',): ('MambaSingleLayer',),
    ('imputation', 'long_term_forecast', 'short_term_forecast'): ('TiDE',),
    ('long_term_forecast',): ('DLinear-paper', 'Koopa', 'MLP7K', 'PatchTST-paper', 'TSBooster', 'TSBooster-constgate', 'TSBooster-learnedgate', 'TSBooster-nogate', 'TSBooster-vanilla', 'TSBooster-wodecomp', 'TSBooster-woexpert', 'iTransformer-paper'),
    ('long_term_forecast', 'short_term_forecast'): ('FreTS', 'Mamba', 'MambaSimple', 'MultiPatchFormer', 'PAttn', 'SCINet', 'TSMixer', 'TemporalFusionTransformer', 'TimeXer', 'WPMixer'),
    ('zero_shot_forecast',): ('Chronos', 'Chronos2', 'Moirai', 'Sundial', 'TiRex', 'TimeMoE', 'TimesFM'),
}

_FOUNDATION_DEPS = {"Chronos": "chronos", "Chronos2": "chronos", "TimesFM": "timesfm",
                    "TimeMoE": "transformers", "Sundial": "transformers",
                    "Moirai": "uni2ts", "TiRex": "tirex"}
_SPECIALIZED_DEPS = {"Mamba": ("mamba_ssm",), "MambaSingleLayer": ("mamba_ssm", "einops"),
                     "Reformer": ("reformer_pytorch",),
                     "WPMixer": ("PyWavelets",), "FEDformer": ("scipy", "sympy"),
                     "ETSformer": ("scipy", "einops"), "FiLM": ("scipy",),
                     "Crossformer": ("einops",), "PAttn": ("einops",),
                     "MambaSimple": ("einops",),
                     "MultiPatchFormer": ("einops",), "KANAD": ("einops",),
                     "MSGNet": ("einops",)}
_DECODER_MODELS = {"Transformer", "Informer", "Autoformer", "FEDformer", "TemporalFusionTransformer"}
_TIME_MARK_MODELS = {"TemporalFusionTransformer", "MambaSingleLayer"}
# TS-Booster family and the paper's baselines: forecast one target (MS/S); the wrapper
# reads target_channel_index.
_SINGLE_TARGET_MODELS = {"MLP7K", "DLinear-paper", "PatchTST-paper", "iTransformer-paper"} | {
    name for name in _TASK_GROUPS[('long_term_forecast',)] if name.startswith("TSBooster")}

MODEL_CAPABILITIES = {}
for tasks, names in _TASK_GROUPS.items():
    for name in names:
        foundation = name in _FOUNDATION_DEPS
        decoder = name in _DECODER_MODELS
        mode = "input_channels" if foundation else "model_defined"
        if name in _SINGLE_TARGET_MODELS:
            mode = "single_target"
        dependencies = (_FOUNDATION_DEPS[name],) if foundation else _SPECIALIZED_DEPS.get(name, ())
        MODEL_CAPABILITIES[name] = ModelCapability(tasks, decoder,
            name in _TIME_MARK_MODELS, mode, dependencies, foundation)


def _declared_tasks(nodes, identifier):
    for node in nodes:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == identifier for t in node.targets):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == identifier:
            value = node.value
        else:
            continue
        try:
            tasks = ast.literal_eval(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{identifier} must be a literal collection of task names") from exc
        if not isinstance(tasks, (tuple, list, set, frozenset)) or not tasks or any(not isinstance(t, str) or t not in _VALID_TASKS for t in tasks):
            raise ValueError(f"Invalid {identifier}: {tasks!r}")
        return tuple(sorted(set(tasks)))
    return None


def _dispatch_tasks(test):
    if isinstance(test, ast.BoolOp):
        return set().union(*(_dispatch_tasks(value) for value in test.values))
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return set()
    left = test.left
    if not isinstance(left, ast.Attribute) or left.attr != "task_name":
        return set()
    try:
        value = ast.literal_eval(test.comparators[0])
    except (ValueError, TypeError):
        return set()
    if isinstance(test.ops[0], ast.Eq) and isinstance(value, str):
        return {value} & _VALID_TASKS
    if isinstance(test.ops[0], ast.In) and isinstance(value, (tuple, list, set)) and all(isinstance(t, str) for t in value):
        return set(value) & _VALID_TASKS
    return set()


def _infer_forward_tasks(model):
    forward = next((node for node in model.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "forward"), None)
    if forward is None:
        return ()
    supported = set()
    for node in ast.walk(forward):
        if not isinstance(node, ast.If):
            continue
        tasks = _dispatch_tasks(node.test)
        # Only accept dispatch branches that actually return a non-None value.
        returns = [n for statement in node.body for n in ast.walk(statement)
                   if isinstance(n, ast.Return) and n.value is not None
                   and not (isinstance(n.value, ast.Constant) and n.value.value is None)]
        if returns and not any(isinstance(statement, ast.Raise) for statement in node.body):
            supported.update(tasks)
    return tuple(sorted(supported))


def get_model_capability(model_name):
    if model_name in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model_name]
    candidates = [p for p in MODEL_ROOT.rglob("*.py") if p.stem == model_name and not p.name.startswith("_")]
    if len(candidates) != 1:
        raise ValueError(f"Unknown or ambiguous model capability: {model_name!r}")
    tree = ast.parse(candidates[0].read_text(encoding="utf-8"))
    model = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model"), None)
    if model is None:
        raise ValueError(f"Model {model_name!r} must define class Model")
    tasks = (_declared_tasks(tree.body, "SUPPORTED_TASKS") or _declared_tasks(model.body, "supported_tasks")
             or _infer_forward_tasks(model))
    if not tasks:
        raise ValueError(f"Model {model_name!r} must declare literal SUPPORTED_TASKS or Model.supported_tasks")
    return ModelCapability(tasks, evidence="source AST declaration or forward dispatch")


def validate_model_task(model_name, task_name):
    capability = get_model_capability(model_name)
    if task_name not in capability.supported_tasks:
        raise ValueError(f"Model {model_name!r} does not implement {task_name!r}; "
                         f"supported tasks: {', '.join(capability.supported_tasks)}")
    return capability
