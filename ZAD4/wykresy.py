"""Rysunki do sprawozdania z zadania 4 (czyta wyniki.json i obrazy z images/ oraz odtworzone/)."""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from PIL import Image

D = json.load(open("wyniki.json"))
C = {"zelda": "#2a78d6", "cameraman": "#eb6834", "peppers": "#1baf7a"}
MARK = {"zelda": "o", "cameraman": "s", "peppers": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd8"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.5, "pdf.fonttype": 42,
})
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def cr_psnr(fname):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, d in D["main"].items():
        K = list(d)
        x = [d[k]["PSNR"] for k in K]; y = [d[k]["CR"] for k in K]
        xe = [d[k]["PSNR_sd"] for k in K]; ye = [d[k]["CR_sd"] for k in K]
        ax.errorbar(x, y, xerr=xe, yerr=ye, color=C[name], marker=MARK[name], ms=5, mec="white", mew=0.7,
                    elinewidth=0.8, capsize=2, ls="--", label=name)
        if name == "zelda":
            for k, xi, yi in zip(K, x, y):
                off = (2, -12) if k == "256" else (4, 5)
                ax.annotate(f"K={k}", (xi, yi), xytext=off, textcoords="offset points", fontsize=7.5,
                            color=C[name])
    ax.set_xlabel("PSNR [dB]"); ax.set_ylabel("współczynnik kompresji CR")
    ax.xaxis.set_major_formatter(COMMA); ax.yaxis.set_major_formatter(COMMA)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


def alive(fname):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for name, d in D["main"].items():
        K = np.array([int(k) for k in d])
        a = np.array([d[k]["neurons"] for k in d]); e = np.array([d[k]["neurons_sd"] for k in d])
        ax.errorbar(K, a, yerr=e, color=C[name], marker=MARK[name], ms=5, mec="white", mew=0.7,
                    elinewidth=0.8, capsize=2, label=name)
    ax.set_xscale("log", base=2); ax.set_xticks(K, [str(k) for k in K])
    ax.set_xlabel("początkowa liczba neuronów K"); ax.set_ylabel("neurony żywe po treningu")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


def variants(fname):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    series = [("frames", "2", "ramka 2×2, wagi losowe (Alg. 2.1)", C["zelda"], "o", "-"),
              ("frames", "4", "ramka 4×4, wagi losowe", C["cameraman"], "s", "-"),
              ("frames", "8", "ramka 8×8, wagi losowe", C["peppers"], "^", "-"),
              ("data_init", "2", "ramka 2×2, wagi startowe z obrazu", C["zelda"], "o", ":"),
              ("data_init", "4", "ramka 4×4, wagi startowe z obrazu", C["cameraman"], "s", ":")]
    for key, n, label, col, m, ls in series:
        d = D[key][n]
        x = [d[k]["PSNR"] for k in d]; y = [d[k]["CR"] for k in d]
        ax.plot(x, y, color=col, marker=m, ms=5, mec="white", mew=0.7, ls=ls, label=label,
                mfc=col if ls == "-" else "white")
    ax.set_yscale("log")
    ax.set_yticks([2, 3, 5, 10, 20, 50], ["2", "3", "5", "10", "20", "50"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("PSNR [dB]"); ax.set_ylabel("CR (skala logarytmiczna)")
    ax.xaxis.set_major_formatter(COMMA)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


def gallery(fname, rows, cols, titles):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.55 * len(cols), 1.62 * len(rows) + 0.2))
    axes = np.atleast_2d(axes)
    for i, name in enumerate(rows):
        for j, src in enumerate(cols):
            ax = axes[i, j]
            path = f"images/{name}.png" if src == "org" else f"odtworzone/{name}_{src}.png"
            ax.imshow(np.asarray(Image.open(path).convert("L")), cmap="gray", vmin=0, vmax=255,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title(titles[j], fontsize=8, color=INK)
            if j == 0 and len(rows) > 1:
                ax.set_ylabel(name, fontsize=8.5, color=INK)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.97)); fig.savefig(fname, dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    cr_psnr("fig_cr_psnr.pdf")
    alive("fig_zywe.pdf")
    variants("fig_warianty.pdf")
    gallery("fig_odtworzone.pdf", ["zelda", "cameraman", "peppers"], ["org", "n2_K4", "n2_K16", "n2_K256"],
            ["oryginał", "K = 4", "K = 16", "K = 256"])
    gallery("fig_ramki.pdf", ["zelda"], ["n2_K256", "n4_K256", "n8_K256", "n2_K256_data", "n4_K256_data"],
            ["2×2, losowe", "4×4, losowe", "8×8, losowe", "2×2, z obrazu", "4×4, z obrazu"])
