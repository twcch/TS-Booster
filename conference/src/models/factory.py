"""Model factory for creating Vanilla / TSB-Lite pairs."""

from src.models.layers import (
    LinearBackbone, LSTMBackbone, GRUBackbone, TransformerBackbone,
)
from src.models.ts_booster import TSBoosterLite


BACKBONES = {
    "linear": LinearBackbone,
    "lstm": LSTMBackbone,
    "gru": GRUBackbone,
    "transformer": TransformerBackbone,
}


def make_backbone(backbone_type, seq_len, input_channels, pred_len, hidden_dim=64, num_layers=2):
    """Create a backbone model by type name."""
    backbone_type = backbone_type.lower()
    if backbone_type == "linear":
        return LinearBackbone(seq_len, input_channels, pred_len)
    elif backbone_type == "lstm":
        return LSTMBackbone(seq_len, input_channels, pred_len, hidden_dim, num_layers)
    elif backbone_type == "gru":
        return GRUBackbone(seq_len, input_channels, pred_len, hidden_dim, num_layers)
    elif backbone_type == "transformer":
        return TransformerBackbone(seq_len, input_channels, pred_len, hidden_dim, num_layers)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def create_model_pair(config, seq_len, pred_len, input_channels):
    """Creates (Vanilla backbone, TSB-Lite wrapped) pair."""
    backbone_type = config.get("backbone_type", "linear")

    vanilla = make_backbone(backbone_type, seq_len, input_channels, pred_len)

    boosted = TSBoosterLite(
        base_model=make_backbone(backbone_type, seq_len, input_channels, pred_len),
        seq_len=seq_len,
        pred_len=pred_len,
        input_channels=input_channels,
        decomp_kernel=config.get("decomp_kernel", 7),
        expert_kernels=config.get("expert_kernels", [3, 5, 7]),
        expert_hidden_dim=config.get("expert_hidden_dim", 32),
        dropout_p=config.get("dropout", 0.3),
    )

    return vanilla, boosted
