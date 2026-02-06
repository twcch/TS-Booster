import copy
from src.models.layers import LSTMBackbone, GRUBackbone, TransformerBackbone, LinearBackbone
from src.models.ts_booster import TS_Booster
from src.models.network import EnhancedDLinear # Keep DLinear legacy access if needed

def create_model_pair(config, seq_len, pred_len, input_channels):
    """
    Creates a pair of models: (Vanilla, Boosted) sharing the same config.
    """
    backbone_type = config.get('backbone_type', 'lstm').lower()
    hidden_dim = config.get('backbone_hidden_dim', 64)
    num_layers = config.get('backbone_num_layers', 2)
    
    # 1. Create Vanilla Model (Backbone Only)
    if backbone_type == 'lstm':
        vanilla_model = LSTMBackbone(seq_len, input_channels, pred_len, hidden_dim, num_layers)
    elif backbone_type == 'gru':
        vanilla_model = GRUBackbone(seq_len, input_channels, pred_len, hidden_dim, num_layers)
    elif backbone_type == 'transformer':
        # Simplify params for factory
        vanilla_model = TransformerBackbone(seq_len, input_channels, pred_len, d_model=hidden_dim, num_layers=num_layers)
    elif backbone_type == 'linear':
        vanilla_model = LinearBackbone(seq_len, input_channels, pred_len)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
        
    # 2. Create Boosted Model
    if backbone_type == 'lstm':
        base_for_booster = LSTMBackbone(seq_len, input_channels, pred_len, hidden_dim, num_layers)
    elif backbone_type == 'gru':
        base_for_booster = GRUBackbone(seq_len, input_channels, pred_len, hidden_dim, num_layers)
    elif backbone_type == 'transformer':
        base_for_booster = TransformerBackbone(seq_len, input_channels, pred_len, d_model=hidden_dim, num_layers=num_layers)
    elif backbone_type == 'linear':
        base_for_booster = LinearBackbone(seq_len, input_channels, pred_len)
        
    boosted_model = TS_Booster(
        base_model=base_for_booster,
        seq_len=seq_len,
        pred_len=pred_len,
        input_channels=input_channels,
        trend_dnas_kernels=config.get('trend_dnas_kernels', [3, 5, 7, 9]),
        seasonal_dnas_kernels=config.get('seasonal_dnas_kernels', [3, 5, 7, 9]),
        decomp_dnas_kernels=config.get('decomp_dnas_kernels', [7, 13, 25, 49, 'identity']),
        trend_hidden_dim=config.get('trend_hidden_dim', 32),
        seasonal_hidden_dim=config.get('seasonal_hidden_dim', 32),
        dropout_p=config.get('trend_dropout', 0.3)
    )
    
    return vanilla_model, boosted_model
