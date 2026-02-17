from src.models.layers import LSTMBackbone, GRUBackbone, TransformerBackbone, LinearBackbone
from src.models.ts_booster import TS_Booster
from src.models.flexible_booster import FlexibleTSBooster

def create_model_pair(config, seq_len, pred_len, input_channels):
    """Creates a pair of models: (Vanilla, Boosted) sharing the same config."""
    backbone_type = config.get('backbone_type', 'lstm').lower()
    hidden_dim = config.get('backbone_hidden_dim', 64)
    num_layers = config.get('backbone_num_layers', 2)

    def make_backbone():
        if backbone_type == 'lstm':
            return LSTMBackbone(seq_len, input_channels, pred_len, hidden_dim, num_layers)
        elif backbone_type == 'gru':
            return GRUBackbone(seq_len, input_channels, pred_len, hidden_dim, num_layers)
        elif backbone_type == 'transformer':
            return TransformerBackbone(seq_len, input_channels, pred_len, hidden_dim=hidden_dim, num_layers=num_layers)
        elif backbone_type == 'linear':
            return LinearBackbone(seq_len, input_channels, pred_len)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    vanilla_model = make_backbone()

    boosted_model = TS_Booster(
        base_model=make_backbone(),
        seq_len=seq_len,
        pred_len=pred_len,
        input_channels=input_channels,
        decomp_kernels=config.get('decomp_kernels', [7, 13]),
        expert_kernels=config.get('expert_kernels', [3, 5, 7]),
        expert_hidden_dim=config.get('expert_hidden_dim', 32),
        dropout_p=config.get('dropout', 0.3),
        n_freq=config.get('n_freq', 4),
        n_boost_rounds=config.get('n_boost_rounds', 2),
        expert_type=config.get('expert_type', 'cnn'),
        expert_gate=config.get('expert_gate', 'snr'),
        enrich_mode=config.get('enrich_mode', 'trend'),
    )

    return vanilla_model, boosted_model
