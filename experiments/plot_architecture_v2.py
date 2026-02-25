"""
TS-Booster Architecture Diagram — Publication Quality with 3D Blocks.
Inspired by deep learning paper architecture figures (ResNet, Transformer style).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np

# ============================================================
# 3D Block Drawing Utilities
# ============================================================

def draw_3d_block(ax, x, y, w, h, d, face_color, edge_color='#37474F',
                  alpha=0.95, label=None, label_size=9, label_color='#212121',
                  sub_label=None, sub_label_size=7, label_bold=True,
                  depth_direction='right'):
    """
    Draw a 3D rectangular block (parallelepiped).
    x, y: bottom-left of front face
    w: width, h: height, d: depth offset
    """
    r, g, b = int(face_color[1:3], 16)/255, int(face_color[3:5], 16)/255, int(face_color[5:7], 16)/255

    # Front face
    front = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                    closed=True, facecolor=face_color, edgecolor=edge_color,
                    linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(front)

    # Top face (lighter)
    top_color = (min(r+0.15, 1), min(g+0.15, 1), min(b+0.15, 1))
    if depth_direction == 'right':
        top = Polygon([(x, y+h), (x+w, y+h), (x+w+d, y+h+d*0.6), (x+d, y+h+d*0.6)],
                      closed=True, facecolor=top_color, edgecolor=edge_color,
                      linewidth=1.0, alpha=alpha, zorder=2)
    else:
        top = Polygon([(x, y+h), (x+w, y+h), (x+w-d, y+h+d*0.6), (x-d, y+h+d*0.6)],
                      closed=True, facecolor=top_color, edgecolor=edge_color,
                      linewidth=1.0, alpha=alpha, zorder=2)
    ax.add_patch(top)

    # Side face (darker)
    side_color = (max(r-0.08, 0), max(g-0.08, 0), max(b-0.08, 0))
    if depth_direction == 'right':
        side = Polygon([(x+w, y), (x+w+d, y+d*0.6), (x+w+d, y+h+d*0.6), (x+w, y+h)],
                       closed=True, facecolor=side_color, edgecolor=edge_color,
                       linewidth=1.0, alpha=alpha, zorder=2)
    else:
        side = Polygon([(x, y), (x-d, y+d*0.6), (x-d, y+h+d*0.6), (x, y+h)],
                       closed=True, facecolor=side_color, edgecolor=edge_color,
                       linewidth=1.0, alpha=alpha, zorder=2)
    ax.add_patch(side)

    # Label
    if label:
        weight = 'bold' if label_bold else 'normal'
        ly = y + h/2 + 0.05 if sub_label else y + h/2
        ax.text(x + w/2, ly, label, ha='center', va='center',
                fontsize=label_size, fontweight=weight, color=label_color, zorder=5)
    if sub_label:
        ax.text(x + w/2, y + h/2 - 0.25, sub_label, ha='center', va='center',
                fontsize=sub_label_size, color='#757575', zorder=5, fontstyle='italic')


def draw_tensor_block(ax, x, y, w, h, d, color, label=None, dim_label=None,
                      label_size=8, dim_size=7):
    """Draw a tall/thin 3D tensor block with dimension annotation."""
    draw_3d_block(ax, x, y, w, h, d, color, label=label, label_size=label_size,
                  label_bold=False)
    if dim_label:
        ax.text(x + w/2, y - 0.2, dim_label, ha='center', va='top',
                fontsize=dim_size, color='#9E9E9E', fontstyle='italic')


def draw_arrow(ax, x1, y1, x2, y2, color='#546E7A', lw=1.8, style='-|>',
               connectionstyle='arc3,rad=0', zorder=4, linestyle='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=connectionstyle,
                                linestyle=linestyle),
                zorder=zorder)


def draw_circle_op(ax, x, y, r, symbol, bg_color='#ECEFF1', edge_color='#37474F',
                   font_size=13):
    circle = plt.Circle((x, y), r, facecolor=bg_color, edgecolor=edge_color,
                        linewidth=1.5, zorder=6)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=font_size,
            fontweight='bold', zorder=7, color='#212121')


def draw_dashed_box(ax, x, y, w, h, label=None, color='#90A4AE', label_size=8):
    """Draw a dashed grouping box."""
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    facecolor='none', edgecolor=color,
                                    linewidth=1.5, linestyle='--', zorder=1)
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h + 0.15, label, ha='center', va='bottom',
                fontsize=label_size, color=color, fontweight='bold')


# ============================================================
# Main Architecture Diagram
# ============================================================

def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.0, 9.5)
    ax.axis('off')
    ax.set_facecolor('white')

    # Color Palette
    C_INPUT = '#90CAF9'      # blue
    C_BACKBONE = '#FFB74D'   # orange
    C_DECOMP = '#81C784'     # green
    C_TREND = '#A5D6A7'      # light green
    C_SEASONAL = '#C8E6C9'   # very light green
    C_CONCAT = '#FFCC80'     # light orange
    C_EXPERT = '#EF9A9A'     # red/pink
    C_GATE = '#CE93D8'       # purple
    C_OUTPUT = '#FFF176'     # yellow
    C_SUM = '#B0BEC5'        # grey

    # ============================================================
    # INPUT BLOCK
    # ============================================================
    draw_3d_block(ax, 0.3, 3.5, 1.2, 2.0, 0.4, C_INPUT,
                  label='Input', label_size=11,
                  sub_label='(B, 30, 2)')

    # Dimension annotation
    ax.text(0.9, 2.9, 'Volatility + Log Return', ha='center', va='top',
            fontsize=7, color='#1565C0', fontstyle='italic')

    # ============================================================
    # PATH 1: BACKBONE (upper path)
    # ============================================================
    # Backbone block
    draw_3d_block(ax, 3.3, 6.5, 2.5, 1.4, 0.5, C_BACKBONE,
                  label='Backbone', label_size=11,
                  sub_label='(Linear / LSTM / ...)')

    # y_base tensor
    draw_tensor_block(ax, 6.8, 6.6, 0.8, 1.2, 0.3, '#FFE082',
                      label='$y_{base}$', label_size=10, dim_label='(B, 3)')

    # Input → Backbone
    draw_arrow(ax, 1.9, 5.2, 3.3, 7.2, connectionstyle='arc3,rad=-0.15',
               color='#1565C0')
    ax.text(2.2, 6.5, 'raw x', fontsize=8, color='#1565C0', fontstyle='italic',
            rotation=30)

    # Backbone → y_base
    draw_arrow(ax, 5.8, 7.2, 6.8, 7.2)

    # ============================================================
    # PATH 2: DECOMPOSITION
    # ============================================================
    # MultiScale Decomp block
    draw_3d_block(ax, 3.3, 3.3, 2.5, 1.8, 0.5, C_DECOMP,
                  label='MultiScale', label_size=10,
                  sub_label='Decomp [7, 13]')

    # Learnable weights annotation
    ax.text(4.55, 3.05, '2 learnable softmax weights', ha='center', va='top',
            fontsize=6.5, color='#2E7D32', fontstyle='italic')

    # Input → Decomp
    draw_arrow(ax, 1.9, 4.5, 3.3, 4.3)

    # Trend output tensor
    draw_tensor_block(ax, 6.6, 4.3, 0.6, 1.5, 0.25, C_TREND,
                      label='trend', label_size=8, dim_label='(B, 30, 2)')

    # Seasonal output tensor
    draw_tensor_block(ax, 6.6, 1.6, 0.6, 1.3, 0.25, C_SEASONAL,
                      label='seas.', label_size=8, dim_label='(B, 30, 2)')

    # Decomp → trend
    draw_arrow(ax, 5.8, 4.6, 6.6, 5.0, connectionstyle='arc3,rad=-0.05')

    # Decomp → seasonal
    draw_arrow(ax, 5.8, 3.8, 6.6, 2.5, connectionstyle='arc3,rad=0.15')

    # ============================================================
    # CONCAT [x, trend]
    # ============================================================
    draw_circle_op(ax, 8.0, 4.7, 0.3, '⊕', bg_color='#E0E0E0')

    # raw x → concat (from input, going under decomp)
    draw_arrow(ax, 1.5, 3.5, 1.5, 4.7, color='#1565C0',
               connectionstyle='arc3,rad=0')
    draw_arrow(ax, 1.5, 4.7, 7.7, 4.7, color='#1565C0',
               connectionstyle='arc3,rad=0')
    ax.text(4.5, 5.0, 'raw x', fontsize=8, color='#1565C0', fontstyle='italic')

    # trend → concat
    draw_arrow(ax, 7.45, 5.05, 7.75, 4.85, connectionstyle='arc3,rad=0.05')

    # Enriched tensor after concat
    draw_tensor_block(ax, 8.7, 4.0, 0.9, 1.8, 0.35, C_CONCAT,
                      label='enriched', label_size=7.5)
    ax.text(9.15, 3.6, '[x, trend]', ha='center', va='top',
            fontsize=7, color='#E65100', fontstyle='italic')
    ax.text(9.15, 3.3, '(B, 30, 4)', ha='center', va='top',
            fontsize=6.5, color='#9E9E9E', fontstyle='italic')

    # concat → enriched
    draw_arrow(ax, 8.3, 4.7, 8.7, 4.9)

    # ============================================================
    # MULTI-ROUND EXPERT (stacked blocks)
    # ============================================================
    # Expert Round 1
    draw_3d_block(ax, 10.2, 5.2, 1.8, 1.0, 0.4, C_EXPERT,
                  label='Expert₁', label_size=9,
                  sub_label='CNN')

    # Expert Round 2
    draw_3d_block(ax, 10.2, 3.7, 1.8, 1.0, 0.4, '#E57373',
                  label='Expert₂', label_size=9,
                  sub_label='CNN')

    # Dashed box around experts
    draw_dashed_box(ax, 10.0, 3.45, 2.45, 3.1,
                    label='Multi-Round Boosting (n=2)', color='#C62828',
                    label_size=8)

    # enriched → Expert 1
    draw_arrow(ax, 9.95, 5.3, 10.2, 5.7, connectionstyle='arc3,rad=-0.1')
    # enriched → Expert 2
    draw_arrow(ax, 9.95, 4.5, 10.2, 4.2, connectionstyle='arc3,rad=0.1')

    # Expert outputs sum
    draw_circle_op(ax, 13.1, 4.7, 0.3, 'Σ', bg_color=C_SUM)

    # Expert 1 → Σ
    draw_arrow(ax, 12.4, 5.7, 12.85, 4.9, connectionstyle='arc3,rad=0.1')
    # Expert 2 → Σ
    draw_arrow(ax, 12.4, 4.2, 12.85, 4.55, connectionstyle='arc3,rad=-0.1')

    # y_expert annotation
    ax.text(13.1, 4.1, '$y_{expert}$', ha='center', va='top',
            fontsize=9, color='#C62828', fontweight='bold')

    # ============================================================
    # SNR GATE
    # ============================================================
    draw_3d_block(ax, 8.3, 1.2, 2.2, 1.2, 0.4, C_GATE,
                  label='SNR Gate', label_size=10, label_color='#4A148C')
    # Formula inside
    ax.text(9.4, 1.35, r'$\sigma(w \cdot \frac{\mathrm{std}_s}{\mathrm{std}_x} + b)$',
            ha='center', va='center', fontsize=8, color='#6A1B9A', zorder=5)

    # "2 params" annotation
    ax.text(9.4, 0.85, 'only 2 learnable params: w, b', ha='center', va='top',
            fontsize=6.5, color='#7B1FA2', fontstyle='italic')

    # seasonal → SNR gate
    draw_arrow(ax, 7.45, 2.0, 8.3, 1.8, color='#2E7D32',
               connectionstyle='arc3,rad=0.05')
    ax.text(7.7, 2.25, '$std_s$', fontsize=8, color='#2E7D32')

    # raw x → SNR gate
    draw_arrow(ax, 1.5, 3.5, 1.5, 1.8, color='#1565C0')
    draw_arrow(ax, 1.5, 1.8, 8.3, 1.8, color='#1565C0')
    ax.text(4.5, 1.5, '$std_x$', fontsize=8, color='#1565C0', fontstyle='italic')

    # Gate multiply node
    draw_circle_op(ax, 13.1, 1.8, 0.3, '×', bg_color='#E1BEE7', edge_color='#4A148C')

    # SNR gate → multiply
    draw_arrow(ax, 10.5, 1.8, 12.8, 1.8, color='#7B1FA2')
    ax.text(11.6, 2.1, 'gate ∈ [0, 1]', fontsize=8, color='#7B1FA2',
            fontstyle='italic')

    # y_expert → multiply
    draw_arrow(ax, 13.1, 4.4, 13.1, 2.1, color='#C62828', lw=1.5)

    # ============================================================
    # FINAL FUSION
    # ============================================================
    draw_circle_op(ax, 14.7, 4.7, 0.35, '+', bg_color='#FFF9C4',
                   edge_color='#F57F17', font_size=16)

    # y_base → final add
    draw_arrow(ax, 7.9, 7.2, 14.4, 4.95, color='#E65100', lw=2.0,
               connectionstyle='arc3,rad=0.1')

    # gated expert → final add
    draw_arrow(ax, 13.1, 1.5, 14.5, 4.4, color='#7B1FA2', lw=2.0,
               connectionstyle='arc3,rad=-0.15')
    ax.text(14.2, 2.8, r'gate $\cdot$ $y_{expert}$', fontsize=8,
            color='#7B1FA2', fontweight='bold', rotation=55)

    # OUTPUT block
    draw_3d_block(ax, 15.4, 4.0, 0.8, 1.4, 0.3, C_OUTPUT,
                  label='$\\hat{y}$', label_size=14, edge_color='#F57F17')
    ax.text(15.8, 3.55, '(B, 3)', ha='center', va='top',
            fontsize=7, color='#9E9E9E', fontstyle='italic')

    # add → output
    draw_arrow(ax, 15.05, 4.7, 15.4, 4.7, color='#F57F17', lw=2.0)

    # ============================================================
    # PATH LABELS
    # ============================================================
    # Path 1 bracket
    ax.annotate('', xy=(2.8, 8.3), xytext=(8.2, 8.3),
                arrowprops=dict(arrowstyle='-', color='#E65100', lw=2.5))
    ax.text(5.5, 8.55, 'Path 1: Base Prediction', ha='center', va='bottom',
            fontsize=10, color='#E65100', fontweight='bold')

    # Path 2 bracket
    ax.annotate('', xy=(2.8, 0.3), xytext=(13.5, 0.3),
                arrowprops=dict(arrowstyle='-', color='#7B1FA2', lw=2.5))
    ax.text(8.0, 0.05, 'Path 2: Decomp-Enriched Expert with SNR-Adaptive Gating',
            ha='center', va='top', fontsize=10, color='#7B1FA2', fontweight='bold')

    # ============================================================
    # FORMULA BOX
    # ============================================================
    formula_text = (
        r'$\hat{y} = y_{\mathrm{base}} + \mathrm{gate} \cdot y_{\mathrm{expert}}$'
        r'$\qquad\qquad$'
        r'$\mathrm{gate} = \sigma\!\left(w \cdot '
        r'\frac{\mathrm{std}(\mathrm{seasonal})}{\mathrm{std}(x)} + b\right)$'
        r'$\qquad\qquad$'
        r'$y_{\mathrm{expert}} = \sum_{i=1}^{n} \mathrm{Expert}_i\!\left([x,\, \mathrm{trend}]\right)$'
    )
    ax.text(8.0, -0.6, formula_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA',
                      edgecolor='#BDBDBD', linewidth=1.2),
            color='#212121')

    # ============================================================
    # TITLE
    # ============================================================
    ax.text(8.0, 9.2, 'TS-Booster: Decomposition-Enriched Expert Boosting with SNR-Adaptive Gating',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#212121')

    # ============================================================
    # LEGEND
    # ============================================================
    legend_x = 0.0
    legend_y = 8.8
    legend_items = [
        (C_INPUT, 'Input Tensor'),
        (C_BACKBONE, 'Backbone'),
        (C_DECOMP, 'Decomposition'),
        (C_EXPERT, 'Expert (CNN)'),
        (C_GATE, 'SNR Gate'),
        (C_OUTPUT, 'Output'),
    ]
    for i, (color, name) in enumerate(legend_items):
        lx = legend_x + i * 2.1
        rect = mpatches.FancyBboxPatch((lx, legend_y), 0.3, 0.3,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='#37474F',
                                        linewidth=0.8)
        ax.add_patch(rect)
        ax.text(lx + 0.45, legend_y + 0.15, name, fontsize=7, va='center',
                color='#424242')

    plt.tight_layout()
    plt.savefig('experiments/comprehensive_results/architecture/ts_booster_architecture_v2.png',
                dpi=250, bbox_inches='tight', facecolor='white')
    plt.savefig('experiments/comprehensive_results/architecture/ts_booster_architecture_v2.pdf',
                bbox_inches='tight', facecolor='white')
    print("Saved to experiments/comprehensive_results/architecture/ts_booster_architecture_v2.png/.pdf")
    plt.close()


if __name__ == "__main__":
    draw_architecture()
