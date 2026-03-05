import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import os

from src.utils.metrics import HybridDirectionalLoss
from src.engine.adaptive_scheduler import AdaptiveWeightScheduler


def train_v11(
    train_loader,
    val_loader,
    test_loader,
    device,
    horizon,
    config=None,
    seq_len=60,
    num_epochs=120,
    lr=0.001,
    weight_decay=1e-4,
    verbose=True,
    # Legacy params
    trendCNNExpert_KernelSize=None,
    seasonalCNNExpert_KernelSize=None,
    seriesDecomposition_KernelSize=None,
    model_hyperparams=None,
    model=None,
):
    if config is None:
        config = {}
    if model_hyperparams:
        config.update(model_hyperparams)

    active_lr = config.get('lr', lr)
    active_weight_decay = config.get('weight_decay', weight_decay)
    active_epochs = config.get('epochs', num_epochs)

    if model is None:
        raise ValueError("Model must be provided via 'model' parameter")
    model = model.to(device)

    dir_weight = config.get('dir_weight', 0.5)
    criterion = HybridDirectionalLoss(dir_weight=dir_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=active_lr, weight_decay=active_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=0.001, min_lr=1e-5
    )

    best_val_loss = float("inf")
    patience = 30
    counter = 0

    # Adaptive weight scheduler for base/expert balance
    warmup_epochs = config.get('dnas_warmup_epochs', max(1, active_epochs // 5))
    adaptive_scheduler = AdaptiveWeightScheduler(warmup_epochs=warmup_epochs, verbose=verbose)

    checkpoint_file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.pth', delete=False)
    checkpoint_path = checkpoint_file.name
    checkpoint_file.close()

    if verbose:
        print("\n[Training] TS-Booster...")
        print("=" * 60)
        print("  Active Configuration:")
        for k, v in config.items():
            print(f"    {k:<30}: {v}")
        print("=" * 60)
        print(f"  Training for {active_epochs} epochs with LR={active_lr:.6f}")
        print()

    for epoch in range(active_epochs):

        # === 1. Training ===
        model.train()
        losses = []
        trend_gate_values = []
        seas_gate_values = []
        aux_losses = []

        for batch in train_loader:
            x, y = batch["raw_input"].to(device), batch["target"].to(device)
            optimizer.zero_grad()
            prev_val = x[:, -1, 0:1]

            model_output = model(x)
            out = model_output[0]
            y_base_out = model_output[1]
            gate_weights = model_output[2]
            y_expert_out = model_output[5] if len(model_output) > 5 else None

            loss = criterion(out, y, prev_val)

            # Adaptive multi-objective loss (base + expert balance)
            if y_expert_out is not None and y_base_out is not None:
                residual_target = y - y_base_out.detach()
                aux_loss = F.mse_loss(y_expert_out, residual_target)

                base_loss = F.mse_loss(y_base_out, y)

                if config.get('fixed_loss_weights'):
                    aux_weight = config.get('fixed_aux_weight', 0.3)
                    base_weight = config.get('fixed_base_weight', 0.1)
                else:
                    weights = adaptive_scheduler.get_weights(epoch)
                    aux_weight = weights['aux']
                    base_weight = weights['base']

                loss = loss + aux_weight * aux_loss + base_weight * base_loss
                aux_losses.append((aux_loss.item(), aux_weight, base_loss.item(), base_weight))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())

            if gate_weights is not None:
                if len(gate_weights) >= 2:
                    trend_gate_values.append(gate_weights[0].item())
                    seas_gate_values.append(gate_weights[1].item())

        # === 2. Validation ===
        model.eval()
        val_losses = []
        val_base_losses = []
        val_expert_losses = []

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["raw_input"].to(device), batch["target"].to(device)
                prev_val = x[:, -1, 0:1]

                model_output = model(x)
                out = model_output[0]
                y_base_val = model_output[1] if len(model_output) > 1 else None
                y_expert_val = model_output[5] if len(model_output) > 5 else None

                loss = criterion(out, y, prev_val)
                val_losses.append(loss.item())

                if y_base_val is not None:
                    val_base_losses.append(F.mse_loss(y_base_val, y).item())
                if y_expert_val is not None and y_base_val is not None:
                    residual_target_val = y - y_base_val
                    val_expert_losses.append(F.mse_loss(y_expert_val, residual_target_val).item())

        train_loss = np.mean(losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_base = np.mean(val_base_losses) if val_base_losses else avg_val_loss
        avg_val_expert = np.mean(val_expert_losses) if val_expert_losses else avg_val_loss

        adaptive_scheduler.update(avg_val_base, avg_val_expert, avg_val_loss, epoch)
        scheduler.step(avg_val_loss)

        t_w = np.mean(trend_gate_values) if trend_gate_values else 0.0
        s_w = np.mean(seas_gate_values) if seas_gate_values else 0.0

        # === 3. Logging ===
        current_lr = optimizer.param_groups[0]["lr"]

        if aux_losses and len(aux_losses[0]) == 4:
            avg_aux_loss = np.mean([x[0] for x in aux_losses])
            avg_aux_weight = np.mean([x[1] for x in aux_losses])
            avg_base_loss_train = np.mean([x[2] for x in aux_losses])
            avg_base_weight = np.mean([x[3] for x in aux_losses])
        else:
            avg_aux_loss = np.mean([x[0] for x in aux_losses]) if aux_losses else 0.0
            avg_aux_weight = np.mean([x[1] for x in aux_losses]) if aux_losses else 0.0
            avg_base_loss_train = 0.0
            avg_base_weight = 0.0

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            if aux_losses and len(aux_losses[0]) == 4:
                aux_info = f" | Exp: {avg_aux_loss:.3f}(w={avg_aux_weight:.2f}) | Base: {avg_base_loss_train:.3f}(w={avg_base_weight:.2f})"
            else:
                aux_info = f" | Aux: {avg_aux_loss:.4f}(w={avg_aux_weight:.2f})" if aux_losses else ""

            print(
                f"Epoch {epoch + 1:3d} | LR: {current_lr:.6f} | Train: {train_loss:.4f} | Val: {avg_val_loss:.4f}{aux_info} | T_Gate: {t_w:.3f} | S_Gate: {s_w:.3f}"
            )

            # MultiScale Decomp weights logging
            if hasattr(model, 'get_multiscale_weights'):
                ms_weights = model.get_multiscale_weights()
                if ms_weights:
                    sorted_items = sorted(ms_weights.items(), key=lambda item: item[1], reverse=True)
                    weights_str = "[" + ", ".join([f"{k}={v:.2f}" for k, v in sorted_items]) + "]"
                    print(f"    [MultiScale Decomp] Weights: {weights_str}")
            print("-" * 60)

        # === 4. Early Stopping ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"[Early Stopping] No improvement for {patience} epochs. Stopped at epoch {epoch + 1}.")
                break

    # === 5. Restore Best Model ===
    if verbose:
        print(f"\n[Training Completed] Loading best model (Val Loss: {best_val_loss:.4f})...")

        if hasattr(model, 'experts'):
            summary = adaptive_scheduler.get_summary()
            print("\n[Adaptive Loss Summary]")
            print(f"  Best Losses - Overall: {summary['best_overall']:.4f}, Base: {summary['best_base']:.4f}, Expert: {summary['best_expert']:.4f}")
            print(f"  Final Weights - Aux: {summary['final_aux_weight']:.3f}, Base: {summary['final_base_weight']:.3f}")
            print(f"  Average Weights - Aux: {summary['avg_aux_weight']:.3f}, Base: {summary['avg_base_weight']:.3f}")

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    try:
        os.unlink(checkpoint_path)
    except:
        pass

    return model


def train_baseline(
    train_loader,
    val_loader,
    test_loader,
    device,
    model,
    num_epochs=100,
    lr=1e-3,
    verbose=True,
):
    """Clean training loop for non-TSB baseline models.

    Uses standard MSE loss + Adam optimizer (matches original papers for
    DLinear, PatchTST, iTransformer, TimeMixer).
    """
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=0.001, min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience = 30
    counter = 0

    checkpoint_file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.pth', delete=False)
    checkpoint_path = checkpoint_file.name
    checkpoint_file.close()

    if verbose:
        print(f"\n[Training Baseline] {model.__class__.__name__}  LR={lr}")

    for epoch in range(num_epochs):
        # Train
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["raw_input"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad()
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["raw_input"].to(device)
                y = batch["target"].to(device)
                out = model(x)
                pred = out[0] if isinstance(out, tuple) else out
                val_losses.append(criterion(pred, y).item())

        train_loss = np.mean(losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d} | LR: {current_lr:.6f} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"  [Early Stop] epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    try:
        os.unlink(checkpoint_path)
    except:
        pass

    if verbose:
        print(f"  Best val loss: {best_val_loss:.4f}")

    return model
