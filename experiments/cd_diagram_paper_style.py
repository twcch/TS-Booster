"""Regenerate the paper's critical-difference diagrams in the existing style.

Style constants reverse-engineered from figures/fig_cd_dl.pdf (4 methods,
2 label rows, 3 cliques) so a new 4-method diagram is pixel-consistent with
its sibling figures.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

AXIS_C, LEAD_C, BAR_C = '#0b0b0b', '#898781', '#52514e'
TSB_C, CD_C = '#2a78d6', '#e34948'
FS = 8.5
Q05 = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}

# --- geometry in axes fraction, taken from fig_cd_dl.pdf -------------------
X_LO, X_HI = 0.22, 0.78          # rank 1 .. rank k
Y_AXIS = 0.53155
TICK_H = 0.04053
Y_TICKLBL = 0.61079              # tick-label baseline
ROW_Y = [0.33333, 0.18919]       # leader/label rows, top to bottom
LEAD_L, LEAD_R = 0.20501, 0.79501
TXT_L, TXT_R = 0.20000, 0.80000
BAR_Y0, BAR_DY = 0.44146, 0.049555
BAR_PAD = 0.0062
Y_CD, Y_CDLBL = 0.83770, 0.89440
FIG_W, FIG_H = 494.740324 / 72, 174.71952 / 72
AX_RECT = [39.1657002 / 494.740324, 7.2 / 174.71952,
           448.374624 / 494.740324, 160.31952 / 174.71952]


def ranks_from(rmse):
    """rmse: dict method -> list of 16 values. Returns mean ranks (ties averaged)."""
    ms = list(rmse)
    M = np.array([rmse[m] for m in ms]).T          # [16 x k]
    R = np.array([stats.rankdata(row) for row in M])
    return ms, R.mean(0), R


def friedman_no_tie(R):
    n, k = R.shape
    a = R.mean(0)
    return 12 * n / (k * (k + 1)) * (np.sum(a ** 2) - k * (k + 1) ** 2 / 4)


def cliques(order_ranks, cd):
    """Maximal runs of adjacent methods (sorted by rank) within cd of each other."""
    out, k = [], len(order_ranks)
    for i in range(k):
        j = i
        while j + 1 < k and order_ranks[j + 1] - order_ranks[i] < cd:
            j += 1
        if j > i:
            out.append((i, j))
    # drop runs contained in an earlier run
    return [b for a, b in enumerate(out)
            if not any(o[0] <= b[0] and b[1] <= o[1] for o in out[:a])]


def draw(methods, avg, cd, path, bold='TS-Booster+Linear'):
    k = len(methods)
    order = np.argsort(avg)
    names = [methods[i] for i in order]
    rr = [avg[i] for i in order]

    def X(r):
        return X_LO + (r - 1) / (k - 1) * (X_HI - X_LO)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(AX_RECT)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.plot([X_LO, X_HI], [Y_AXIS] * 2, color=AXIS_C, lw=1.2,
            solid_capstyle='projecting', clip_on=False)
    for t in range(1, k + 1):
        ax.plot([X(t)] * 2, [Y_AXIS, Y_AXIS + TICK_H], color=AXIS_C, lw=1.0,
                solid_capstyle='projecting', clip_on=False)
        ax.text(X(t), Y_TICKLBL, str(t), ha='center', va='baseline',
                fontsize=FS, color=AXIS_C)

    half = (k + 1) // 2
    for n, (nm, r) in enumerate(zip(names, rr)):
        left = n < half
        row = n if left else k - 1 - n
        y = ROW_Y[row]
        ex = LEAD_L if left else LEAD_R
        ax.plot([X(r)] * 2, [Y_AXIS, y], color=LEAD_C, lw=0.8, clip_on=False)
        ax.plot([ex, X(r)], [y] * 2, color=LEAD_C, lw=0.8, clip_on=False)
        mc = TSB_C if nm == bold else AXIS_C
        ax.plot([X(r)], [Y_AXIS], 'o', ms=4, mfc=mc, mec=mc,
                mew=1, clip_on=False)
        lab = f'{nm} ({r:.2f})' if left else f'({r:.2f}) {nm}'
        ax.text(TXT_L if left else TXT_R, y, lab,
                ha='right' if left else 'left', va='center', fontsize=FS,
                color=TSB_C if nm == bold else AXIS_C,
                fontweight='bold' if nm == bold else 'normal')

    for b, (i, j) in enumerate(cliques(rr, cd)):
        ax.plot([X(rr[i]) - BAR_PAD, X(rr[j]) + BAR_PAD], [BAR_Y0 - b * BAR_DY] * 2,
                color=BAR_C, lw=3.2, alpha=0.85, solid_capstyle='round', clip_on=False)

    x0 = X(1) + (X(2) - X(1)) * 0.025 if k > 1 else X_LO
    x1 = x0 + (X_HI - X_LO) / (k - 1) * cd
    ax.plot([x0, x1], [Y_CD] * 2, color=CD_C, lw=1.3, clip_on=False)
    for xe in (x0, x1):
        ax.plot([xe] * 2, [Y_CD - 0.014, Y_CD + 0.014], color=CD_C, lw=1.3, clip_on=False)
    ax.text((x0 + x1) / 2, Y_CDLBL, f'CD = {cd:.2f}', ha='center', va='baseline',
            fontsize=FS, color=CD_C, fontweight='bold')

    fig.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return names, rr


if __name__ == '__main__':
    # Statistical group restricted to same-target comparators (paper Fig. 3).
    # Per-dataset RMSE from the paper's Table 4, in the dataset order of Table 1.
    TSB = [1.5282, 1.8049, 2.0504, 1.5986, 1.6950, 1.6839, 1.7736, 1.3999,
           1.6570, 1.9459, 7.5071, 6.6660, 5.8629, 7.5864, 10.2181, 5.2176]
    NAIVE = [2.1350, 2.2251, 2.8460, 2.2135, 2.1727, 2.0439, 2.1843, 1.7356,
             2.0940, 2.5978, 6.8347, 8.3186, 7.9338, 9.7847, 13.7978, 6.8140]
    HAR = [1.8728, 1.9961, 2.5138, 1.9427, 1.9968, 1.8530, 1.9803, 1.6289,
           1.9318, 2.3818, 6.5019, 7.8996, 7.4012, 9.0544, 13.0313, 6.4760]
    ARIMA = [2.0988, 2.0898, 2.6196, 2.1961, 2.1573, 1.9199, 2.1843, 1.6175,
             1.9505, 2.3639, 6.8347, 7.9330, 7.3559, 8.7451, 13.7978, 6.5417]

    ms, avg, R = ranks_from({'TS-Booster+Linear': TSB, 'Persistence': NAIVE,
                             'HAR-RV': HAR, 'ARIMA': ARIMA})
    k = len(ms)
    chi2 = friedman_no_tie(R)                       # no tie correction, as in the paper
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * 16))
    print(f'chi2({k-1}) = {chi2:.2f}  p = {1 - stats.chi2.cdf(chi2, k-1):.3g}  CD = {cd:.4f}')
    for n, a in sorted(zip(ms, avg), key=lambda x: x[1]):
        print(f'  {n:20s} {a:.4f}')
    draw(ms, avg, cd, 'fig_cd_statistical.pdf')
