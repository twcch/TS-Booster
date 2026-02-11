import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import os

from src.models.network import EnhancedDLinear
from src.utils.metrics import HybridDirectionalLoss
from src.engine.adaptive_scheduler import AdaptiveWeightScheduler


def train_v11(
    train_loader,
    val_loader,
    test_loader,
    device,
    horizon,
    config=None,  # 新增: 優先使用的配置字典
    seq_len=60,
    num_epochs=120,   # 預設值 (若 config 沒指定)
    lr=0.001,         # 預設值 (若 config 沒指定)
    weight_decay=1e-4, # 預設值
    verbose=True,
    # 為了兼容舊調用方式，保留這幾個參數，但它們會被 config 覆蓋
    trendCNNExpert_KernelSize=None,
    seasonalCNNExpert_KernelSize=None,
    seriesDecomposition_KernelSize=None,
    model_hyperparams=None,
    model=None, # Allow injecting pre-created model
):
    # 1. 處理配置優先級: Config > Explicit Args > Defaults
    if config is None:
        config = {}
    
    # 處理模型超參數 (合併舊有的 model_hyperparams)
    if model_hyperparams:
        config.update(model_hyperparams)
        
    # 處理 DNAS 相關 override (如果 config 中沒有，則保持預設)
    # 舊的單一 kernel size 參數在這裡不再強制覆蓋 config，以免衝突
    pass

    # 提取訓練參數
    active_lr = config.get('lr', lr)
    active_weight_decay = config.get('weight_decay', weight_decay)
    active_epochs = config.get('epochs', num_epochs) # 允許 config 控制 epoch
    
    # 準備模型參數 (移除訓練相關參數，避免傳入 model.__init__ 報錯)
    training_keys = {'lr', 'weight_decay', 'epochs', 'batch_size', 'backbone_type'} 
    # 注意: backbone_type 在 network.py 是參數之一，不應移除，除非它不是 model param
    # 檢查 network.py, backbone_type 是 __init__ 參數，所以應該保留。
    # 修正: 只移除 model 不接受的 training params
    model_init_keys = set(config.keys()) - {'lr', 'weight_decay', 'epochs', 'batch_size', 'input_channels', 'dnas_warmup_epochs'}
    model_kwargs = {k: config[k] for k in model_init_keys}

    # 確保 DNAS 參數存在
    if 'trend_dnas_kernels' not in model_kwargs:
        model_kwargs['trend_dnas_kernels'] = [3, 5, 7, 9]
    if 'seasonal_dnas_kernels' not in model_kwargs:
        model_kwargs['seasonal_dnas_kernels'] = [3, 5, 7, 9]
    if 'decomp_dnas_kernels' not in model_kwargs:
        model_kwargs['decomp_dnas_kernels'] = [7, 13, 25, 49]

    if model is None:
        model = EnhancedDLinear(
            seq_len=seq_len,
            pred_len=horizon,
            input_channels=config.get('input_channels', 2),
            **model_kwargs
        ).to(device)
    else:
        model = model.to(device)

    criterion = HybridDirectionalLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=active_lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=active_lr, weight_decay=active_weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=active_epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=0.001, min_lr=1e-5
    )

    best_val_loss = float("inf")
    patience = 20
    counter = 0

    # P6: Initialize adaptive weight scheduler
    dnas_warmup_epochs = config.get('dnas_warmup_epochs', max(1, active_epochs // 5))
    adaptive_scheduler = AdaptiveWeightScheduler(warmup_epochs=dnas_warmup_epochs, verbose=verbose)

    # 使用臨時文件避免不同配置之間的衝突
    checkpoint_file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.pth', delete=False)
    checkpoint_path = checkpoint_file.name
    checkpoint_file.close()

    if verbose:
        print("\n[Training] Enhanced DLinear...")
        print("="*60)
        print("  Active Configuration:")
        for k, v in config.items():
            print(f"    {k:<30}: {v}")
        print("="*60)
        print(f"  Training for {active_epochs} epochs with LR={active_lr:.6f}")
        print("\n")
    for epoch in range(active_epochs):
        # === DNAS & Gating Scheduler ===
        # P1 fix: default warmup = 20% of total epochs
        dnas_warmup_epochs = config.get('dnas_warmup_epochs', max(1, active_epochs // 5))

        # Check if model supports DNAS toggling
        if hasattr(model, 'toggle_dnas_mode') and hasattr(model, 'set_gating_status'):
            if epoch < dnas_warmup_epochs:
                if epoch == 0 and verbose: print(f"  [Scheduler] Warmup Mode: DNAS Frozen, Gates Fixed (Epoch 0-{dnas_warmup_epochs-1})")
                model.toggle_dnas_mode("warmup")
                model.set_gating_status(fixed=True)
            else:
                if epoch == dnas_warmup_epochs:
                    if verbose:
                        print(f"  [Scheduler] Search Mode: DNAS Active, Gates Adaptive (Epoch {dnas_warmup_epochs}+)")
                        print(f"  [P6 - Adaptive Multi-Objective] Dynamic loss weighting activated")
                    # Reset early stopping + LR so search gets a fair chance
                    counter = 0
                    for pg in optimizer.param_groups:
                        pg['lr'] = active_lr
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.5, patience=10, threshold=0.001, min_lr=1e-5
                    )
                    if verbose: print(f"  [LR Reset] LR restored to {active_lr}, scheduler patience=10")
                model.toggle_dnas_mode("search")
                model.set_gating_status(fixed=False)

                # P0 fix: Tau annealing during search phase
                if hasattr(model, 'set_tau'):
                    search_epochs = active_epochs - dnas_warmup_epochs
                    search_progress = (epoch - dnas_warmup_epochs) / max(1, search_epochs - 1)
                    tau = max(0.1, 1.0 - search_progress * 0.9)
                    model.set_tau(tau)
                    if verbose and epoch == dnas_warmup_epochs:
                        print(f"  [Tau Annealing] tau: 1.0 → 0.1 over {search_epochs} search epochs")

        # === 1. Training ===
        model.train()
        losses = []
        trend_gate_values = []
        seas_gate_values = []
        aux_losses = []  # Track aux loss separately for diagnostics

        for batch in train_loader:
            x, y = batch["raw_input"].to(device), batch["target"].to(device)
            optimizer.zero_grad()
            prev_val = x[:, -1, 0:1]

            # Unpack model output (5 or 6 values)
            model_output = model(x)
            out = model_output[0]
            y_base_out = model_output[1]
            gate_weights = model_output[2]
            y_expert_out = model_output[5] if len(model_output) > 5 else None

            loss = criterion(out, y, prev_val)

            # P6: Adaptive Multi-Objective Loss
            # Compute auxiliary losses for base and expert separately
            if y_expert_out is not None and y_base_out is not None:
                # Expert auxiliary loss: how well expert predicts residual
                residual_target = y - y_base_out.detach()
                aux_loss = F.mse_loss(y_expert_out, residual_target)

                # Base auxiliary loss: how well base predicts by itself
                base_loss = F.mse_loss(y_base_out, y)

                # Get adaptive weights from scheduler
                # (Weights are updated based on validation performance each epoch)
                weights = adaptive_scheduler.get_weights(epoch)
                aux_weight = weights['aux']
                base_weight = weights['base']

                # Combined loss with adaptive weights
                loss = loss + aux_weight * aux_loss + base_weight * base_loss
                aux_losses.append((aux_loss.item(), aux_weight, base_loss.item(), base_weight))  # Track for logging

            # Entropy penalty: encourage DNAS alpha to be decisive (not uniform)
            if epoch >= dnas_warmup_epochs and hasattr(model, 'get_arch_parameters'):
                arch_params = list(model.get_arch_parameters())
                if arch_params:
                    search_epochs = active_epochs - dnas_warmup_epochs
                    search_progress = (epoch - dnas_warmup_epochs) / max(1, search_epochs - 1)
                    entropy_weight = max(0.05, 0.3 * (1 - search_progress))

                    entropy_loss = 0.0
                    for alpha in arch_params:
                        probs = F.softmax(alpha, dim=0)
                        entropy_loss += -(probs * torch.log(probs + 1e-8)).sum()
                    loss = loss + entropy_weight * entropy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())

            # 紀錄 Gate 權重 (如果有回傳)
            if gate_weights is not None:
                # [Trend_Gate, Seasonal_Gate, Max_Trend_DNAS, Max_Seasonal_DNAS]
                if len(gate_weights) >= 4:
                    trend_gate_values.append(gate_weights[0].item())
                    seas_gate_values.append(gate_weights[1].item())
                    # 可以選擇性紀錄 max_dnas，這裡暫時只記錄 Gate
                else:
                    trend_gate_values.append(gate_weights[0].item())
                    seas_gate_values.append(gate_weights[1].item())

        # === 2. Validation ===
        # P6: Compute separate losses for base, expert, and overall
        model.eval()
        val_losses = []
        val_base_losses = []
        val_expert_losses = []

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["raw_input"].to(device), batch["target"].to(device)
                prev_val = x[:, -1, 0:1]

                # Get full model output
                model_output = model(x)
                out = model_output[0]  # Overall prediction
                y_base_val = model_output[1] if len(model_output) > 1 else None
                y_expert_val = model_output[5] if len(model_output) > 5 else None

                # Overall loss
                loss = criterion(out, y, prev_val)
                val_losses.append(loss.item())

                # Separate component losses for adaptive weighting
                if y_base_val is not None:
                    base_loss_val = F.mse_loss(y_base_val, y)
                    val_base_losses.append(base_loss_val.item())

                if y_expert_val is not None and y_base_val is not None:
                    residual_target_val = y - y_base_val
                    expert_loss_val = F.mse_loss(y_expert_val, residual_target_val)
                    val_expert_losses.append(expert_loss_val.item())

        train_loss = np.mean(losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_base = np.mean(val_base_losses) if val_base_losses else avg_val_loss
        avg_val_expert = np.mean(val_expert_losses) if val_expert_losses else avg_val_loss

        # P6: Update adaptive scheduler with component losses
        adaptive_scheduler.update(avg_val_base, avg_val_expert, avg_val_loss, epoch)

        scheduler.step(avg_val_loss)

        # 計算平均權重 (若無 CNN 則為 0)
        t_w = np.mean(trend_gate_values) if trend_gate_values else 0.0
        s_w = np.mean(seas_gate_values) if seas_gate_values else 0.0

        # === 3. Logging & Early Stopping ===
        # 印出當前狀態 (包含權重資訊)
        # === 3. Logging & Early Stopping ===
        # 印出當前狀態 (包含權重資訊)
        current_lr = optimizer.param_groups[0]["lr"]
        # === 3. Logging & Early Stopping ===
        # 印出當前狀態 (包含權重資訊)
        current_lr = optimizer.param_groups[0]["lr"]

        # Calculate average aux_loss and weight for this epoch
        # P6: Now tracking both aux (expert) and base losses
        if aux_losses and len(aux_losses[0]) == 4:  # New format with base loss
            avg_aux_loss = np.mean([x[0] for x in aux_losses])
            avg_aux_weight = np.mean([x[1] for x in aux_losses])
            avg_base_loss_train = np.mean([x[2] for x in aux_losses])
            avg_base_weight = np.mean([x[3] for x in aux_losses])
        else:  # Fallback for old format
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
            
            # --- DNAS Diagnostics Logging ---
            # --- DNAS Diagnostics Logging ---
            if hasattr(model, 'get_dnas_status'):
                dnas_status = model.get_dnas_status()
                
                def format_probs(probs):
                    if isinstance(probs, dict):
                        sorted_items = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                        return "[" + ", ".join([f"{k}={v:.2f}" for k, v in sorted_items]) + "]"
                    elif isinstance(probs, list):
                        return "[" + ", ".join([f"{p:.2f}" for p in probs]) + "]"
                    return str(probs)

                if "decomp" in dnas_status:
                    print(f"    [DNAS] Decomp Weights: {format_probs(dnas_status['decomp'])}")
                if "trend_expert" in dnas_status: # Update for booster key name
                     print(f"    [DNAS] Trend Expert: {format_probs(dnas_status['trend_expert'])}")
                elif "trend_cnn" in dnas_status:
                    print(f"    [DNAS] Trend CNN Weights: {format_probs(dnas_status['trend_cnn'])}")
                    
                if "seasonal_expert" in dnas_status: # Update for booster key name
                     print(f"    [DNAS] Seasonal Expert: {format_probs(dnas_status['seasonal_expert'])}")
                elif "seasonal_cnn" in dnas_status:
                    print(f"    [DNAS] Seasonal CNN Weights: {format_probs(dnas_status['seasonal_cnn'])}")
                print("-" * 60)

        # Checkpoint logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # 儲存最佳模型參數到臨時文件
            torch.save(model.state_dict(), checkpoint_path)
            # print("  >>> New Best Model Saved!") # 可選擇是否要在每個變好時都印
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(
                        f"[Early Stopping] No improvement for {patience} epochs. Stopped at epoch {epoch + 1}."
                    )
                break

    # === 4. Restore Best Model ===
    if verbose:
        print(
            f"\n[Training Completed] Loading best model (Val Loss: {best_val_loss:.4f})..."
        )

        # P6: Print adaptive scheduler summary
        if hasattr(model, 'toggle_dnas_mode'):  # Only for models with DNAS
            summary = adaptive_scheduler.get_summary()
            print("\n[P6.1 - Adaptive Multi-Objective Summary (with Smoothing)]")
            print(f"  Best Losses - Overall: {summary['best_overall']:.4f}, Base: {summary['best_base']:.4f}, Expert: {summary['best_expert']:.4f}")
            print(f"  Final Weights - Aux: {summary['final_aux_weight']:.3f}, Base: {summary['final_base_weight']:.3f}")
            print(f"  Average Weights - Aux: {summary['avg_aux_weight']:.3f}, Base: {summary['avg_base_weight']:.3f}")
            print(f"  Final Smoothed Gaps - Base: {summary['final_smoothed_base_gap']:.4f}, Expert: {summary['final_smoothed_expert_gap']:.4f}")

    # 從臨時文件載入最佳模型
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    # 清理臨時文件
    try:
        os.unlink(checkpoint_path)
    except:
        pass

    return model
