"""Non-neural comparators on the framework's exact windows (paper Table 5 and Eq. 19).

  Persistence         y_hat(t+h) = y_t, the last observed target
  ObservedWindowRule  Eq. 19: std(r_{t+h-w+1}, ..., r_t; ddof=1) * sqrt(A) * S, the observed part of
                      the rolling target window (needs w - h >= 2)
  EWMA                RiskMetrics variance of the returns (lambda 0.94); flat single-day volatility
  HAR-RV              OLS of y_{t+h} on [1, y_t, 5-day and 22-day means of y], one fit per step on the
                      training origins
  GARCH, GJR-GARCH    (1,1) fitted by arch on the training returns in percent; analytic h-step
                      variance from the filtered state at t, annualized
  ARIMA               pmdarima auto_arima on the training and validation targets, updated with each
                      observed test value
  XGBoost, LightGBM   one regressor per step on the flattened scaled window, early-stopped on the
                      validation windows and seeded by --rand_seed
  EWMA-aligned,       the same variance models forecasting the rolling target itself (reviewer 2.5):
  GARCH-aligned,      the window's known returns plus model variances of its future returns give
  GJR-GARCH-aligned   E[S^2] of the target's sample variance, and the forecast is sqrt(E[S^2] * A) * S

Settings are the paper's (Section 4.1.4) and fixed here; effective_config.json lists them.
EWMA and the GARCH models forecast single-day conditional volatility rather than the
rolling target, so the paper reports them as reference comparisons; the -aligned
variants are a revision control for that mismatch. The statistical models read the
whole series up to the forecast origin, not only the look-back window.

The CLI is run.py's, so run_batch_long_term_forecast.py schedules these like the neural
models, as long_term_forecast runs that fit on the training split (is_training 1).
Results use the standard artifact layout read by tools/parse_result_to_xlsx.py and tools/stats.

    python tools/baselines/classical_baselines.py --task_name long_term_forecast --is_training 1 \\
      --model HAR-RV --model_id USD_TWD --data custom --root_path dataset/volatility \\
      --data_path USD_TWD.csv --features MS --target Volatility --seq_len 30 --label_len 0 --pred_len 3 --inverse
"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run  # noqa: E402
from data_provider.data_factory import data_provider  # noqa: E402
from data_provider.preparation import read_forecast_csv, validate_forecast_frame  # noqa: E402
from exp.common.evaluation import metric_spaces, process_peak_rss_mb, run_summary  # noqa: E402
from exp.common.runtime import (atomic_save_array, mark_run_failed, prepare_run, resolve_device,  # noqa: E402
                                save_evaluation)

EWMA_LAMBDA = 0.94
HAR_WEEK, HAR_MONTH = 5, 22
RETURN_PERCENT = 100.0  # arch fits percent returns
ARIMA_SEARCH = {"seasonal": False, "stepwise": True, "suppress_warnings": True, "error_action": "ignore",
                "max_order": 10, "max_p": 5, "max_q": 5}
TREES = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}
EARLY_STOPPING = 30
SETTINGS = {
    "Persistence": {"forecast": "last observed target"},
    "ObservedWindowRule": {"forecast": "Eq. 19, the std of the known returns in the target window"},
    "EWMA": {"lambda": EWMA_LAMBDA, "initial_variance": "r_0^2", "forecast": "flat single-day volatility"},
    "HAR-RV": {"regressors": ["1", "y_t", f"mean of {HAR_WEEK} days", f"mean of {HAR_MONTH} days"],
               "fit": f"OLS per step on training origins t >= {HAR_MONTH}"},
    "GARCH": {"p": 1, "o": 0, "q": 1, "mean": "constant", "dist": "normal", "fit": "training returns (percent)"},
    "GJR-GARCH": {"p": 1, "o": 1, "q": 1, "mean": "constant", "dist": "normal",
                  "fit": "training returns (percent)"},
    "ARIMA": {"search": ARIMA_SEARCH, "fit": "training and validation targets",
              "update": "after each observed test value"},
    "XGBoost": {**TREES, "early_stopping_rounds": EARLY_STOPPING},
    # LightGBM ignores subsample without subsample_freq, so it does not subsample rows (as in the paper).
    "LightGBM": {**TREES, "early_stopping_rounds": EARLY_STOPPING, "row_subsampling": False},
    "EWMA-aligned": {"lambda": EWMA_LAMBDA, "forecast": "rolling target from known returns and EWMA variances",
                     "future_returns": "zero mean, uncorrelated"},
    "GARCH-aligned": {"p": 1, "o": 0, "q": 1, "mean": "constant", "dist": "normal", "fit": "training returns (percent)",
                      "forecast": "rolling target from known returns and h-step GARCH variances",
                      "future_returns": "zero mean, uncorrelated"},
    "GJR-GARCH-aligned": {"p": 1, "o": 1, "q": 1, "mean": "constant", "dist": "normal",
                          "fit": "training returns (percent)",
                          "forecast": "rolling target from known returns and h-step GJR-GARCH variances",
                          "future_returns": "zero mean, uncorrelated"},
}
DEPENDENCIES = {"GARCH": "arch", "GJR-GARCH": "arch", "ARIMA": "pmdarima", "XGBoost": "xgboost",
                "LightGBM": "lightgbm", "GARCH-aligned": "arch", "GJR-GARCH-aligned": "arch"}
ALIGNED = {"EWMA-aligned", "GARCH-aligned", "GJR-GARCH-aligned"}
NEEDS_RETURNS = {"ObservedWindowRule", "EWMA", "GARCH", "GJR-GARCH"} | ALIGNED
SEEDED = {"XGBoost", "LightGBM"}


def build_parser():
    parser = run.build_parser()
    parser.description = __doc__.split("\n\n")[0]
    parser.add_argument("--vol_window", type=int, default=7,
                        help="rolling target window w (ObservedWindowRule and the -aligned models)")
    parser.add_argument("--annualization", type=float, default=252.0, help="A (ObservedWindowRule, EWMA, GARCH)")
    parser.add_argument("--vol_scale", type=float, default=100.0, help="S (ObservedWindowRule, EWMA, GARCH)")
    parser.add_argument("--return_column", default="log_return",
                        help="return feature of ObservedWindowRule, EWMA and the GARCH models")
    return parser


class BaselineModel(torch.nn.Module):
    """Parameter-free stand-in so run identity and artifacts follow the framework."""

    def __init__(self, effective_config):
        super().__init__()
        self.effective_config = effective_config


class Panel:
    """One run's data: raw values, split boundaries and the scaled windows of each split."""

    def __init__(self, args):
        self.args = args
        self.splits = {flag: data_provider(args, flag)[0] for flag in ("train", "val", "test")}
        self.test = self.splits["test"]
        path = Path(args.root_path) / args.data_path
        if not path.is_file():
            raise ValueError(f"classical baselines read a local CSV; not found: {path}")
        _, names, self.values = validate_forecast_frame(read_forecast_csv(path), args.target, args.features)
        if names != self.test.feature_names:
            raise ValueError("CSV columns differ from the dataset's features")
        first, _ = self.test.split_bounds
        self.n_train, self.n_val = self.test.train_boundary, first + args.seq_len
        if self.test.scale and not np.array_equal(self.test.scaler.transform(self.values[first:]),
                                                  self.test.data_x):
            raise ValueError("raw values do not reproduce the dataset's test split")
        self.target = self.test.target_index
        self.order = [self.target] + [i for i in range(len(names)) if i != self.target]
        self.test_x, self.targets = self.windows("test")
        self.anchors = self.test_x[:, -1, self.target]
        self.origins = first + args.seq_len - 1 + np.arange(len(self.test_x))
        self.steps = np.arange(args.pred_offset + 1, args.pred_offset + args.pred_len + 1)

    def windows(self, flag):
        dataset = self.splits[flag]
        pairs = [dataset[index][:2] for index in range(dataset.base_length)]
        x = np.stack([np.asarray(window) for window, _ in pairs])
        y = np.stack([np.asarray(target)[-self.args.pred_len:, self.target] for _, target in pairs])
        return x, y

    def raw(self, column=None):
        return self.values[:, self.target if column is None else self.test.feature_names.index(column)]

    def to_scaled(self, values):
        if not self.test.scale:
            return values
        return (values - self.test.scaler.mean_[self.target]) / self.test.scaler.scale_[self.target]

    def flat(self, x):
        """Tree features: per step [target, other channels], flattened step by step."""
        return x[:, :, self.order].reshape(len(x), -1)


def persistence(panel, args):
    return np.repeat(panel.anchors[:, None], args.pred_len, axis=1), {}, 0


def observed_window(panel, args):
    known = [args.vol_window - int(h) for h in panel.steps]
    if min(known) < 2 or max(known) > args.seq_len:
        raise ValueError(f"Eq. 19 needs 2 <= w - h <= seq_len for steps {panel.steps.tolist()} "
                         f"(w={args.vol_window})")
    returns = panel.raw(args.return_column)
    rule = np.array([[np.std(returns[t - k + 1:t + 1], ddof=1) for k in known] for t in panel.origins])
    return panel.to_scaled(rule * np.sqrt(args.annualization) * args.vol_scale), {
        "formula": "y_hat(t+h) = std(r_{t+h-w+1}, ..., r_t; ddof=1) * sqrt(A) * S",
        "vol_window": args.vol_window, "known_returns_per_step": known}, 0


def _ewma_variance(returns):
    """RiskMetrics variance; index t holds the forecast sigma^2_{t+1|t}."""
    variance = np.empty(len(returns))
    value = variance[0] = returns[0] ** 2
    for t in range(1, len(returns)):
        value = EWMA_LAMBDA * value + (1 - EWMA_LAMBDA) * returns[t] ** 2
        variance[t] = value
    return variance


def ewma(panel, args):
    variance = _ewma_variance(panel.raw(args.return_column))
    sigma = np.sqrt(variance[panel.origins] * args.annualization) * args.vol_scale
    return panel.to_scaled(np.repeat(sigma[:, None], args.pred_len, axis=1)), {}, 0


def _aligned(panel, args, variances):
    """Forecast the rolling target itself from its known returns and model variances of the rest.

    variances(t) gives sigma^2_{t+1|t}, ..., sigma^2_{t+m|t} for decimal returns (m = last step).
    With zero-mean, uncorrelated future returns, the target window's sample variance has
    E[S^2] = (sum_K r^2 + sum_F s^2 - ((sum_K r)^2 + sum_F s^2) / w) / (w - 1) over its known (K)
    and future (F) returns; the forecast is sqrt(E[S^2] * A) * S.
    """
    window, returns = args.vol_window, panel.raw(args.return_column)
    predictions = np.zeros((len(panel.origins), args.pred_len))
    for i, t in enumerate(panel.origins):
        future = variances(t)
        for k, h in enumerate(panel.steps):
            first = t + int(h) - window + 1  # first return in the target window
            known = returns[first:t + 1] if first <= t else returns[:0]
            unknown = future[max(first - t, 1) - 1:int(h)].sum()
            expected = (np.square(known).sum() + unknown - (known.sum() ** 2 + unknown) / window) / (window - 1)
            predictions[i, k] = np.sqrt(max(expected, 0.0) * args.annualization) * args.vol_scale
    return panel.to_scaled(predictions)


ALIGNED_FORMULA = ("y_hat(t+h) = sqrt(E[S^2] * A) * S; E[S^2] = (sum_K r^2 + sum_F s^2 - ((sum_K r)^2 + sum_F s^2) / w)"
                   " / (w - 1), K = known returns, F = future returns with model variances s^2")


def ewma_aligned(panel, args):
    variance = _ewma_variance(panel.raw(args.return_column))
    steps = int(panel.steps[-1])
    return _aligned(panel, args, lambda t: np.full(steps, variance[t])), {"formula": ALIGNED_FORMULA}, 0


def har_rv(panel, args):
    from sklearn.linear_model import LinearRegression
    y = panel.raw()

    def regressors(origins):
        return np.array([[1.0, y[t], np.mean(y[t - HAR_WEEK + 1:t + 1]), np.mean(y[t - HAR_MONTH + 1:t + 1])]
                         for t in origins])

    started = time.perf_counter()
    origins = np.arange(HAR_MONTH, panel.n_train)
    design = regressors(origins)
    fits = {int(h): LinearRegression(fit_intercept=False).fit(design, y[origins + h]) for h in panel.steps}
    fit_s = time.perf_counter() - started
    test = regressors(panel.origins)
    predictions = np.stack([fits[int(h)].predict(test) for h in panel.steps], axis=1)
    return panel.to_scaled(predictions), {
        "coefficients": {h: fit.coef_.tolist() for h, fit in fits.items()},
        "training_origins": [int(origins[0]), int(origins[-1])], "fit_time_s": fit_s}, 4 * len(fits)


def _garch_filter(panel, args, asymmetric):
    """Fit (GJR-)GARCH(1,1) on training returns in percent; variance[t] = sigma^2_{t|t-1} past training."""
    from arch import arch_model
    returns = panel.raw(args.return_column) * RETURN_PERCENT
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = arch_model(returns[:panel.n_train], vol="Garch", p=1, o=int(asymmetric), q=1, dist="normal",
                            rescale=False).fit(disp="off", show_warning=False)
    fit_s = time.perf_counter() - started
    omega, alpha = fitted.params["omega"], fitted.params.get("alpha[1]", 0.0)
    gamma, beta = fitted.params.get("gamma[1]", 0.0), fitted.params.get("beta[1]", 0.0)
    # Filter past the training data with the fitted recursion (returns, not residuals).
    variance = np.zeros(len(returns))
    value = float(fitted.conditional_volatility[-1] ** 2)
    shock = float(returns[panel.n_train - 1])
    for t in range(panel.n_train, len(returns)):
        value = max(omega + (alpha + gamma * (1.0 if shock < 0 else 0.0)) * shock ** 2 + beta * value, 1e-8)
        variance[t] = value
        shock = returns[t]
    decay = alpha + 0.5 * gamma + beta  # variance persistence; GJR counts half of gamma
    long_run = omega / max(1 - decay, 1e-8)
    details = {"params": {name: float(value) for name, value in fitted.params.items()},
               "persistence": float(decay), "convergence_flag": int(fitted.convergence_flag), "fit_time_s": fit_s}
    return variance, long_run, decay, details, len(fitted.params)


def garch(panel, args, asymmetric=False):
    variance, long_run, decay, details, params = _garch_filter(panel, args, asymmetric)
    predictions = np.array([[np.sqrt(max(long_run + decay ** int(h) * (variance[t] - long_run), 1e-8)
                                     / RETURN_PERCENT ** 2 * args.annualization) * args.vol_scale
                             for h in panel.steps] for t in panel.origins])
    return panel.to_scaled(predictions), details, params


def garch_aligned(panel, args, asymmetric=False):
    variance, long_run, decay, details, params = _garch_filter(panel, args, asymmetric)
    ahead = np.arange(int(panel.steps[-1]))

    def variances(t):  # sigma^2_{t+j|t} from sigma^2_{t+1|t} = variance[t + 1], which uses r_t
        return np.maximum(long_run + decay ** ahead * (variance[t + 1] - long_run), 1e-8) / RETURN_PERCENT ** 2
    return _aligned(panel, args, variances), {"formula": ALIGNED_FORMULA, **details}, params


def arima(panel, args):
    import pmdarima
    y = panel.raw()
    horizon = args.pred_offset + args.pred_len
    predictions = np.zeros((len(panel.origins), args.pred_len))
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pmdarima.auto_arima(y[:panel.n_val], **ARIMA_SEARCH)
        fit_s = time.perf_counter() - started
        selected = {"order": list(model.order), "with_intercept": bool(model.with_intercept),
                    "aic": float(model.aic()), "params": model.params().tolist()}
        for i in range(len(panel.origins)):
            predictions[i] = model.predict(n_periods=horizon)[-args.pred_len:]
            model.update(y[panel.n_val + i:panel.n_val + i + 1])
    return panel.to_scaled(predictions), {"selected": selected, "fit_time_s": fit_s}, len(selected["params"])


def _tree_data(panel):
    train_x, train_y = panel.windows("train")
    val_x, val_y = panel.windows("val")
    return panel.flat(train_x), train_y, panel.flat(val_x), val_y, panel.flat(panel.test_x)


def _trees(panel, args, fit_step):
    train_x, train_y, val_x, val_y, test_x = _tree_data(panel)
    predictions, trees, predict_s = np.zeros((len(test_x), args.pred_len)), [], 0.0
    started = time.perf_counter()
    for step in range(args.pred_len):
        model, count = fit_step(train_x, train_y[:, step], val_x, val_y[:, step])
        tick = time.perf_counter()
        with warnings.catch_warnings():  # LightGBM names numpy columns, then sklearn warns at predict
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            predictions[:, step] = model.predict(test_x)
        predict_s += time.perf_counter() - tick
        trees.append(count)
    fit_s = time.perf_counter() - started - predict_s
    return predictions, {"trees_per_step": trees, "seed": args.rand_seed, "fit_time_s": fit_s}, None


def xgboost_trees(panel, args):
    from xgboost import XGBRegressor

    def fit_step(x, y, val_x, val_y):
        model = XGBRegressor(**TREES, early_stopping_rounds=EARLY_STOPPING, random_state=args.rand_seed, verbosity=0)
        model.fit(x, y, eval_set=[(val_x, val_y)], verbose=False)
        return model, int(model.best_iteration) + 1
    return _trees(panel, args, fit_step)


def lightgbm_trees(panel, args):
    import lightgbm

    def fit_step(x, y, val_x, val_y):
        model = lightgbm.LGBMRegressor(**TREES, random_state=args.rand_seed, verbose=-1)
        model.fit(x, y, eval_set=[(val_x, val_y)], callbacks=[lightgbm.early_stopping(EARLY_STOPPING, verbose=False),
                                                             lightgbm.log_evaluation(0)])
        return model, int(model.best_iteration_)
    return _trees(panel, args, fit_step)


METHODS = {"Persistence": persistence, "ObservedWindowRule": observed_window, "EWMA": ewma, "HAR-RV": har_rv,
           "GARCH": garch, "GJR-GARCH": lambda panel, args: garch(panel, args, asymmetric=True),
           "ARIMA": arima, "XGBoost": xgboost_trees, "LightGBM": lightgbm_trees,
           "EWMA-aligned": ewma_aligned, "GARCH-aligned": garch_aligned,
           "GJR-GARCH-aligned": lambda panel, args: garch_aligned(panel, args, asymmetric=True)}


def main(argv=None):
    parser = build_parser()
    requested = parser.parse_args(argv)
    if requested.model not in METHODS:
        parser.error(f"--model must be one of {', '.join(METHODS)}")
    if requested.task_name != "long_term_forecast" or requested.is_training != 1:
        parser.error("classical baselines fit on the training split: use --task_name long_term_forecast "
                     "--is_training 1")
    if requested.data != "custom" or requested.features not in {"MS", "S"}:
        parser.error("classical baselines need --data custom and --features MS or S")
    if requested.pred_offset < 0 or not 0 <= requested.label_len <= requested.seq_len:
        parser.error("require pred_offset >= 0 and 0 <= label_len <= seq_len")
    if requested.augmentation_ratio:
        parser.error("classical baselines do not use --augmentation_ratio")
    dependency = DEPENDENCIES.get(requested.model)
    if dependency and importlib.util.find_spec(dependency) is None:
        parser.error(f"{requested.model} needs {dependency}: pip install -r requirements/baselines.txt")
    args = deepcopy(requested)
    args.requested_config = deepcopy(vars(requested))
    args.use_gpu, args.gpu_type, args.use_multi_gpu = False, "cpu", False
    resolve_device(args)
    panel = Panel(args)
    if args.model in NEEDS_RETURNS and args.return_column not in panel.test.feature_names:
        parser.error(f"{args.model} needs the {args.return_column!r} input column (--features MS)")
    args.enc_in = args.dec_in = len(panel.test.feature_names)
    args.c_out = 1
    effective = {"architecture": args.model, "settings": SETTINGS[args.model],
                 "deterministic": args.model not in SEEDED,
                 "target_channel": panel.test.feature_names[panel.target], "features": panel.test.feature_names}
    if args.model in NEEDS_RETURNS:
        effective.update(annualization=args.annualization, vol_scale=args.vol_scale,
                         return_column=args.return_column)
    if args.model == "ObservedWindowRule" or args.model in ALIGNED:
        effective["vol_window"] = args.vol_window
    setting = prepare_run(args, BaselineModel(effective), run.legacy_setting(args, 0), 0)
    try:
        started = time.perf_counter()
        predictions, details, params = METHODS[args.model](panel, args)
        compute_s = time.perf_counter() - started
        fit_s = details.pop("fit_time_s", None)
        forward_s = compute_s - (fit_s or 0.0)
        predictions, targets, metrics, extra = metric_spaces(
            args, panel.test, predictions[..., None], panel.targets[..., None], anchors=panel.anchors[:, None])
        directory = Path(args.results_root) / setting
        atomic_save_array(directory / "last.npy", np.asarray(extra.pop("_anchors")))
        extra.update(baseline={"method": args.model, **details}, additional_artifacts=["last.npy"],
                     train_time_s=fit_s, forward_time_s=forward_s,
                     forward_ms_per_sample=forward_s * 1000 / len(predictions), num_samples=len(predictions),
                     params=params, trainable_params=params, model_size_mb=None,
                     process_peak_rss_mb=process_peak_rss_mb())
        save_evaluation(args, setting, metrics, predictions, targets, extra=extra, summary=run_summary(metrics, extra))
    except BaseException as error:
        mark_run_failed(args, error)
        raise
    print(", ".join(f"{name}:{value}" for name, value in metrics.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
