import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from neuron_liniowy import delta_training, eta_sweep, make_data, run_case

ETA, K, REPEATS = 0.01, 10000, 5
CASES = [(2, 6), (4, 4), (6, 3)]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # stala kolejnosc
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]                 # kodowanie wtorne (druk)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd8"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
})


COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212"))  # polski zapis liczb


def plot_mse(res, fname):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    epochs = np.arange(1, K + 1)
    for i, r in enumerate(res["runs"]):
        ax.plot(epochs, r["mse"], color=COLORS[i], ls=STYLES[i], label=f"wykonanie {i + 1}")
    if res["N"] < res["M"]:
        ax.axhline(res["mse_ls"], color=INK, lw=0.8, ls=(0, (2, 2)))
        ax.annotate("min. MSE (rozw. LS) = " + f"{res['mse_ls']:.4f}".replace(".", ","), xy=(K, res["mse_ls"]),
                    xytext=(-4, 4), textcoords="offset points", ha="right", color=INK, fontsize=9)
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(COMMA)
    else:
        ax.set_yscale("log")
    ax.set_xlabel("epoka")
    ax.set_ylabel("MSE na zbiorze treningowym")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def delta_path(X, Z, w0, eta, K):
    """Trajektoria wag zapisywana po kazdym wzorcu (do ilustracji 2D)."""
    w, path = w0.copy(), [w0.copy()]
    for _ in range(K):
        for mu in range(len(Z)):
            w = w + eta * (Z[mu] - w @ X[mu]) * X[mu]
            path.append(w.copy())
    return np.array(path)


def plot_geometry(fname, seed=18, eta=0.1, K=300, lim=1.3):
    X3, Z3 = make_data(2, 3, seed)
    starts = [np.random.default_rng(100 + r).uniform(-1, 1, 2) for r in range(REPEATS)]
    g = np.linspace(-lim, lim, 300)
    W1, W2 = np.meshgrid(g, g)
    titles = ["$M=1<N$ (nieoznaczony)", "$M=2=N$ (oznaczony)", "$M=3>N$ (sprzeczny)"]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.9), sharey=True)
    for ax, M, title in zip(axes, (1, 2, 3), titles):
        X, Z = X3[:M], Z3[:M]
        E = np.mean([(Z[m] - (X[m, 0] * W1 + X[m, 1] * W2)) ** 2 for m in range(M)], axis=0)
        ax.contour(W1, W2, E, levels=12, colors=GRID, linewidths=0.6)
        for m in range(M):                       # prosta x^m . w = z^m
            a, b = X[m]
            ax.plot(g, (Z[m] - a * g) / b, color=MUTED, lw=0.9)
            inside = np.abs((Z[m] - a * g) / b) < lim - 0.1
            xl = g[inside][int(0.80 * inside.sum())]
            ax.text(xl, (Z[m] - a * xl) / b, rf" $\mu={m + 1}$", color=MUTED, fontsize=7.5, va="bottom")
        for i, w0 in enumerate(starts):
            p = delta_path(X, Z, w0, eta, K)
            ax.plot(p[:, 0], p[:, 1], color=COLORS[i], lw=1.0, ls=STYLES[i])
            ax.plot(*w0, "o", ms=6, mfc="white", mec=COLORS[i], mew=1.5)
            ax.plot(*p[-1], "o", ms=6, color=COLORS[i], mec="white", mew=0.8)
        w_ls = np.linalg.pinv(X) @ Z
        ax.plot(*w_ls, "*", ms=11, color=INK, mec="white", mew=0.6, zorder=5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, color=INK)
        ax.xaxis.set_major_formatter(COMMA); ax.yaxis.set_major_formatter(COMMA)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.set_xlabel("$w_1$")
    axes[0].set_ylabel("$w_2$")
    handles = [plt.Line2D([], [], color=COLORS[i], ls=STYLES[i], label=f"wykonanie {i + 1}") for i in range(REPEATS)]
    handles += [plt.Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, label="wagi pocz."),
                plt.Line2D([], [], ls="", marker="o", color=MUTED, label="wagi końc."),
                plt.Line2D([], [], ls="", marker="*", ms=10, color=INK, label=r"$X^{+}z$")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = {}
    for N, M in CASES:
        res = run_case(N, M, ETA, K, REPEATS)
        plot_mse(res, f"mse_N{N}_M{M}.pdf")
        out[f"N{N}_M{M}"] = dict(
            rank=int(res["rank"]), cond=res["cond"], null_dim=int(res["null_dim"]),
            w_ls=res["w_ls"].tolist(), mse_ls=res["mse_ls"], norm_ls=res["norm_ls"], spread=res["spread"],
            runs=[{k: (v.tolist() if isinstance(v, np.ndarray) and k != "mse" else v)
                   for k, v in r.items() if k != "mse"} for r in res["runs"]],
        )
    # N = M: dluzszy trening (wolny kierunek przy najmniejszej wartosci osobliwej)
    X, Z = make_data(4, 4)
    w_star = np.linalg.solve(X, Z)
    out["NM_long"] = {str(k): max(np.linalg.norm(delta_training(X, Z, ETA, k, rng=np.random.default_rng(100 + r))[1] - w_star)
                                  for r in range(REPEATS)) for k in (10000, 30000)}
    out["NM_sv"] = np.linalg.svd(X, compute_uv=False).tolist()
    out["eta_sweep"] = eta_sweep(2, 6, [0.1, 0.03, 0.01, 0.003, 0.001], 100000)
    plot_geometry("geometria_2d.pdf")
    with open("wyniki.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("OK")
