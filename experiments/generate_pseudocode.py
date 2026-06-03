"""Generate pseudocode as a clean PNG image for paper."""
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'monospace'

lines = [
    ("bold",  "Algorithm 1: TS-Booster Forward Pass"),
    ("line",  ""),
    ("normal", "Input:  x ∈ ℝ^{B × S × C}              ▷ raw time series"),
    ("normal", "Output: y_final ∈ ℝ^{B × P}             ▷ enhanced prediction"),
    ("line",  ""),
    ("gray",  "// Learnable parameters:"),
    ("gray",  "//   decomp_weights ∈ ℝ^K                 (softmax-normalized)"),
    ("gray",  "//   gate_w ∈ ℝ, gate_b ∈ ℝ              (SNR gate)"),
    ("gray",  "//   Expert₁, ..., Expert_R                (R independent networks)"),
    ("line",  ""),
    ("bold",  "function TS_Booster_Forward(x):"),
    ("line",  ""),
    ("gray",  "    // Step 1: Multi-Scale Decomposition"),
    ("normal", "    w ← softmax(decomp_weights)"),
    ("normal", "    trend ← Σ_{k=1}^{K}  w_k · AvgPool1d(x, kernel=k)"),
    ("normal", "    seasonal ← x − trend"),
    ("line",  ""),
    ("gray",  "    // Step 2: Backbone Prediction (unmodified)"),
    ("normal", "    y_base ← Backbone(x)"),
    ("line",  ""),
    ("gray",  "    // Step 3: Trend-Enriched Expert Input"),
    ("normal", "    x_enriched ← concat([x, trend], dim=channel)    ▷ ∈ ℝ^{B × S × 2C}"),
    ("line",  ""),
    ("gray",  "    // Step 4: Multi-Round Boosting"),
    ("normal", "    y_expert ← 0"),
    ("bold",  "    for r = 1 to R do"),
    ("normal", "        y_expert ← y_expert + Expert_r(x_enriched)"),
    ("bold",  "    end for"),
    ("line",  ""),
    ("gray",  "    // Step 5: SNR-Based Adaptive Gating"),
    ("normal", "    snr ← std(seasonal) / std(x)"),
    ("normal", "    gate ← sigmoid(gate_w · snr + gate_b)"),
    ("normal", "    y_expert ← gate · y_expert"),
    ("line",  ""),
    ("gray",  "    // Step 6: Additive Fusion"),
    ("normal", "    y_final ← y_base + y_expert"),
    ("line",  ""),
    ("bold",  "    return y_final"),
]

fig, ax = plt.subplots(figsize=(10, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

n = len(lines)
line_height = 1.0 / (n + 2)
y_start = 1 - line_height

for i, (style, text) in enumerate(lines):
    y = y_start - i * line_height
    if style == "line":
        continue
    elif style == "bold":
        ax.text(0.03, y, text, fontsize=10.5, fontweight="bold",
                fontfamily="monospace", va="top", color="#1a1a1a")
    elif style == "gray":
        ax.text(0.03, y, text, fontsize=10, fontweight="normal",
                fontfamily="monospace", va="top", color="#666666")
    else:
        ax.text(0.03, y, text, fontsize=10, fontweight="normal",
                fontfamily="monospace", va="top", color="#1a1a1a")

# Border
rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98,
                      linewidth=1.5, edgecolor="#333333",
                      facecolor="#fafafa", zorder=-1)
ax.add_patch(rect)

# Top separator line (under title)
ax.plot([0.02, 0.98], [y_start - 0.8 * line_height, y_start - 0.8 * line_height],
        color="#333333", linewidth=1.0)

out = "/home/nush/paper_booster/experiments/comprehensive_results/pseudocode.png"
plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
plt.close()
print(f"Saved to {out}")
