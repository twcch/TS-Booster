import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import os

from src.models.network import EnhancedDLinear
from src.utils.metrics import HybridDirectionalLoss


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
                        print(f"  [P5 - Expert-Aware DNAS] Aux Loss weight boosted: 0.1 → 1.0 (search phase)")
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

            # Auxiliary residual loss: teach expert to learn what backbone can't predict
            # P5 FIX: Expert-aware DNAS - boost aux_loss during search phase
            if y_expert_out is not None and y_base_out is not None:
                residual_target = y - y_base_out.detach()
                aux_loss = F.mse_loss(y_expert_out, residual_target)

                # Dynamic aux_weight based on training phase
                if epoch < dnas_warmup_epochs:
                    # Warmup: moderate weight, let backbone learn first
                    aux_weight = 0.1
                else:
                    # Search phase: HIGH weight to guide DNAS toward expert-friendly decomp
                    search_epochs = active_epochs - dnas_warmup_epochs
                    search_progress = (epoch - dnas_warmup_epochs) / max(1, search_epochs - 1)
                    # Decay from 1.0 → 0.1 (keep high throughout search)
                    aux_weight = max(0.1, 1.0 * (1 - search_progress * 0.9))

                loss = loss + aux_weight * aux_loss
                aux_losses.append((aux_loss.item(), aux_weight))  # Track for logging

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
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["raw_input"].to(device), batch["target"].to(device)
                prev_val = x[:, -1, 0:1]
                out = model(x)[0]
                loss = criterion(out, y, prev_val)
                val_losses.append(loss.item())

        train_loss = np.mean(losses)
        avg_val_loss = np.mean(val_losses)
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
        avg_aux_loss = np.mean([x[0] for x in aux_losses]) if aux_losses else 0.0
        avg_aux_weight = np.mean([x[1] for x in aux_losses]) if aux_losses else 0.0

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
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
    
    # 從臨時文件載入最佳模型
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    # 清理臨時文件
    try:
        os.unlink(checkpoint_path)
    except:
        pass

    return model
