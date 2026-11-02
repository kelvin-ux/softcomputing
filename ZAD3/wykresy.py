import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

D = json.load(open("wyniki.json"))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # neuron / wzorzec 1..4
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd8"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.4, "pdf.fonttype": 42,
})
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def errors(fname):
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for key, color, label in (("bias", C[0], "z biasem"), ("nobias", C[1], "bez biasu")):
        for i, r in enumerate(D[key]):
            h = np.array(r["history"])
            ep = np.arange(len(h)) * 10
            ax.plot(ep[1:], h[1:], color=color, lw=0.8, alpha=0.55, label=label if i == 0 else None)
    ax.axhline(D["target"], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(12, D["target"] * 1.25, "E = 0,01", color=MUTED, fontsize=8)
    ax.axhline(0.125, color=INK, lw=0.8, ls=(0, (1, 1.5)))
    ax.text(12, 0.125 * 1.2, "E = 1/8 (dolne ograniczenie bez biasu)", color=INK, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoka"); ax.set_ylabel("błąd E (wzór 2.1)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


def hidden_space(fname, seed=0):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))
    for ax, key, title in zip(axes, ("bias", "nobias"), ("z biasem", "bez biasu")):
        r = D[key][seed]
        H, W2 = np.array(r["H"]), np.array(r["W2"])
        lim = 1.25 * np.abs(H).max()
        t = np.linspace(-lim, lim, 200)
        for i in range(4):
            a, b = W2[i, 0], W2[i, 1]
            c = W2[i, 2] if key == "bias" else 0.0
            if abs(b) > abs(a):
                ax.plot(t, -(a * t + c) / b, color=C[i], lw=1.2, ls="--")
            else:
                ax.plot(-(b * t + c) / a, t, color=C[i], lw=1.2, ls="--")
        for k in range(4):
            ax.plot(*H[k], "o", ms=9, color=C[k], mec="white", mew=1.0, zorder=5)
            if key == "bias" or np.abs(H[k]).max() >= 1.6:      # punkty przy (0,0) podpisane w przybliżeniu
                ax.annotate(str(k + 1), H[k], xytext=(6, 5), textcoords="offset points", fontsize=9,
                            color=INK, fontweight="bold")
        ax.plot(0, 0, "+", color=MUTED, ms=8)
        if key == "nobias":                                   # przybliżenie okolic punktu (0, 0)
            ins = ax.inset_axes([0.58, 0.04, 0.4, 0.4])
            z = 1.6
            tz = np.linspace(-z, z, 50)
            for i in range(4):
                a, b = W2[i, 0], W2[i, 1]
                if abs(b) > abs(a):
                    ins.plot(tz, -a * tz / b, color=C[i], lw=1.0, ls="--")
                else:
                    ins.plot(-b * tz / a, tz, color=C[i], lw=1.0, ls="--")
            for k in range(4):
                if np.abs(H[k]).max() < z:
                    ins.plot(*H[k], "o", ms=7, color=C[k], mec="white", mew=0.8)
                    ins.annotate(str(k + 1), H[k], xytext=(4, 3), textcoords="offset points", fontsize=8,
                                 fontweight="bold")
            ins.plot(0, 0, "+", color=MUTED, ms=6)
            ins.set_xlim(-z, z); ins.set_ylim(-z, z); ins.set_xticks([]); ins.set_yticks([])
            ins.set_facecolor("white")
            for s in ins.spines.values():
                s.set_visible(True); s.set_color(MUTED)
            ax.indicate_inset_zoom(ins, edgecolor=MUTED)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("$h_1$"); ax.set_title(title, fontsize=9.5, color=INK)
        ax.xaxis.set_major_formatter(COMMA); ax.yaxis.set_major_formatter(COMMA)
    axes[0].set_ylabel("$h_2$")
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


def eta_plot(fname):
    sw = D["eta_sweep"]
    etas = np.array([float(e) for e in sw])
    med = np.array([np.median(sw[e]["epochs"]) for e in sw])
    lo = np.array([min(sw[e]["epochs"]) for e in sw]); hi = np.array([max(sw[e]["epochs"]) for e in sw])
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.vlines(etas, lo, hi, color=C[0], lw=1.2, alpha=0.5)
    ax.plot(etas, med, color=C[0], marker="o", ms=5, mec="white", mew=0.8)
    for e, m, s in zip(etas, med, [sw[k]["success"] for k in sw]):
        ax.annotate(f"{int(m)}" + ("" if s == 20 else f"  ({s}/20)"), (e, m), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(etas, [f"{e:g}".replace(".", ",") for e in etas])
    ax.set_xlabel("współczynnik uczenia $\\eta$"); ax.set_ylabel("epoki do E < 0,01")
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


if __name__ == "__main__":
    errors("fig_bledy.pdf")
    hidden_space("fig_ukryta.pdf")
    eta_plot("fig_eta.pdf")
