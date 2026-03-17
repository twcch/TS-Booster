"""
Simplified training loops for conference version.

- train_booster(): MSE + aux residual loss for TSB-Lite
- train_baseline(): Standard MSE for vanilla / SOTA models
"""

import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import os


def train_booster(
    train_loader, val_loader, device, model, config,
    num_epochs=100, verbose=True,
):
    """
    Train TSB-Lite with simplified multi-objective loss:
      Loss = MSE(y_final, y_true) + aux_weight * MSE(y_expert, y_true - y_base.detach())
    """
    model = model.to(device)

    lr = config.get("lr", 0.001)
    weight_decay = config.get("weight_decay", 1e-4)
    aux_weight = config.get("aux_weight", 0.3)
    epochs = config.get("epochs", num_epochs)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=0.001, min_lr=1e-5
    )

    best_val_loss = float("inf")
    patience = 30
    counter = 0

    checkpoint = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    checkpoint_path = checkpoint.name
    checkpoint.close()

    if verbose:
        print(f"\n[Training] TSB-Lite | LR={lr} | aux_weight={aux_weight}")

    for epoch in range(epochs):
        # Train
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["raw_input"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad()

            y_final, y_base, y_expert = model(x)

            # Main loss
            loss = criterion(y_final, y)

            # Auxiliary: expert learns residual
            residual = y - y_base.detach()
            aux_loss = F.mse_loss(y_expert, residual)
            loss = loss + aux_weight * aux_loss

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
                y_final, _, _ = model(x)
                val_losses.append(criterion(y_final, y).item())

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
    except OSError:
        pass

    if verbose:
        print(f"  Best val loss: {best_val_loss:.4f}")
    return model


def train_baseline(
    train_loader, val_loader, device, model,
    num_epochs=100, lr=1e-3, verbose=True,
):
    """Standard MSE training for vanilla backbones and SOTA baselines."""
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=0.001, min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience = 30
    counter = 0

    checkpoint = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    checkpoint_path = checkpoint.name
    checkpoint.close()

    if verbose:
        print(f"\n[Training] {model.__class__.__name__} | LR={lr}")

    for epoch in range(num_epochs):
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
    except OSError:
        pass

    if verbose:
        print(f"  Best val loss: {best_val_loss:.4f}")
    return model
