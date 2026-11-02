"""Rysunki do sprawozdania (czyta wyniki.json i katalogi z obrazami wygenerowane przez eksperymenty.py)."""
import json
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from PIL import Image

from madaline_ocr import load_patterns

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "wyniki.json", encoding="utf-8"))
LETTERS = string.ascii_lowercase
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd8"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "pdf.fonttype": 42,
})
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def samples(fname, letters="agmr", levels=(0, 10, 30, 50, 90)):
    fig, axes = plt.subplots(len(letters), len(levels), figsize=(6.0, 4.9))
    for i, l in enumerate(letters):
        for j, p in enumerate(levels):
            ax = axes[i, j]
            img = np.asarray(Image.open(HERE / f"test_patterns_{p}" / f"{l}.png").convert("L"))
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(GRID)
            if i == 0:
                ax.set_title(f"{p}%", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"„{l}”", fontsize=10, color=INK, rotation=0, labelpad=14, va="center")
    fig.tight_layout(pad=0.4)
    fig.savefig(fname)
    plt.close(fig)


def similarity(fname):
    W, labels, _ = load_patterns(HERE / "training_patterns")
    S = W @ W.T
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(S, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ticks = range(len(LETTERS))
    ax.set_xticks(ticks, LETTERS, fontsize=8.5); ax.set_yticks(ticks, LETTERS, fontsize=8.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.yaxis.set_major_formatter(COMMA); cb.outline.set_visible(False)
    cb.set_label("iloczyn skalarny (cos kąta)")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    off = S - np.eye(len(S)) * 2
    pairs = sorted({tuple(sorted((LETTERS[i], LETTERS[j]))): S[i, j]
                    for i in range(26) for j in range(26) if i != j}.items(), key=lambda t: -t[1])
    return pairs[:8], pairs[-5:], float(np.mean(off.max(axis=1)))


def accuracy(fname):
    sw = D["sweep"]
    p = np.array([int(k) for k in sw])
    acc = np.array([sw[k]["acc"] for k in sw]) * 100
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.axhline(100 / 26, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(1, 100 / 26 + 2.5, "zgadywanie (1/26)", ha="left", color=MUTED, fontsize=8)
    ax.plot(p, acc, color=BLUE, marker="o", ms=5, mec="white", mew=0.8)
    for x, y in zip(p, acc):
        if 60 <= x <= 90:
            ax.annotate(f"{y:.1f}%".replace(".", ","), (x, y), xytext=(5, 4), textcoords="offset points",
                        fontsize=8, color=INK)
    ax.set_xlabel("poziom zaszumienia [%]"); ax.set_ylabel("skuteczność rozpoznawania [%]")
    ax.set_xticks(p); ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def outputs(fname):
    sw = D["sweep"]
    p = np.array([int(k) for k in sw])
    corr = np.array([sw[k]["corr"] for k in sw]); sd = np.array([sw[k]["corr_sd"] for k in sw])
    wrong = np.array([sw[k]["wrong"] for k in sw]); theory = np.array([sw[k]["theory"] for k in sw])
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.fill_between(p, corr - sd, corr + sd, color=BLUE, alpha=0.15, lw=0)
    ax.plot(p, corr, color=BLUE, marker="o", ms=4.5, mec="white", mew=0.8, label="neuron właściwej litery")
    ax.plot(p, wrong, color=ORANGE, ls="--", marker="s", ms=4, mec="white", mew=0.8,
            label="najsilniejszy neuron innej litery")
    ax.plot(p, theory, color=INK, ls="", marker="x", ms=5, mew=1.1, label="wartość teoretyczna (wzór 3)")
    ax.set_xlabel("poziom zaszumienia [%]"); ax.set_ylabel("wyjście neuronu (cos kąta)")
    ax.set_xticks(p); ax.yaxis.set_major_formatter(COMMA); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


if __name__ == "__main__":
    samples("fig_probki.pdf")
    top, bottom, mean_max = similarity("fig_podobienstwo.pdf")
    accuracy("fig_skutecznosc.pdf")
    outputs("fig_wyjscia.pdf")
    print("najbardziej podobne:", [(a, round(v, 3)) for a, v in top])
    print("najmniej podobne:", [(a, round(v, 3)) for a, v in bottom])
    print("średnie max podobieństwo do innej litery:", round(mean_max, 3))
