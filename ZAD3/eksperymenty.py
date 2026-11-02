import json

import numpy as np
from scipy.optimize import minimize

from mlp import X, Z, train

SEEDS = range(20)
ETA, EPOCHS = 0.5, 20000
ETAS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
TARGET = 0.01                       # próg błędu E uznany za skuteczny trening


def run_many(bias):
    """20 treningów z różnych wag początkowych."""
    runs = []
    for seed in SEEDS:
        net, hist = train(bias=bias, eta=ETA, epochs=EPOCHS, seed=seed, every=10)
        Y = net.outputs()
        hit = np.flatnonzero(hist < TARGET)
        runs.append(dict(
            seed=seed, E=float(hist[-1]), maxerr=float(np.max(np.abs(Z - Y))),
            argmax_ok=int(np.sum(np.argmax(Y, axis=1) == np.arange(4))),
            epochs_to_target=int(hit[0] * 10) if hit.size else None,
            history=hist.tolist(), Y=Y.tolist(), H=net.hidden_codes().tolist(), W2=net.W2.tolist(),
        ))
    return runs


def eta_sweep():
    """Liczba epok potrzebna do E < TARGET w zależności od eta (z biasem)."""
    out = {}
    for eta in ETAS:
        ep, ok = [], 0
        for seed in SEEDS:
            net, hist = train(bias=True, eta=eta, epochs=EPOCHS, seed=seed, target=TARGET)
            if hist[-1] < TARGET:
                ok += 1
                ep.append(len(hist) - 1)
        out[str(eta)] = dict(success=ok, epochs=ep)
    return out


def best_without_bias(starts=300):
    """Najmniejszy błąd E bez biasu znaleziony innym optymalizatorem (L-BFGS) - niezależnie od propagacji wstecznej."""
    def E(p):
        W1, W2 = p[:8].reshape(2, 4), p[8:].reshape(4, 2)
        S = np.clip((W2 @ W1 @ X.T).T, -500, 500)
        return 0.5 * np.sum((Z - 1 / (1 + np.exp(-S))) ** 2)
    rng = np.random.default_rng(0)
    return float(min(minimize(E, rng.uniform(-3, 3, 16), method="L-BFGS-B").fun for _ in range(starts)))


if __name__ == "__main__":
    res = dict(eta=ETA, epochs=EPOCHS, target=TARGET)
    res["bias"] = run_many(True)
    res["nobias"] = run_many(False)
    for key in ("bias", "nobias"):
        E = [r["E"] for r in res[key]]
        print(f"{key}: E min/med/max {min(E):.4f}/{np.median(E):.4f}/{max(E):.4f}; "
              f"skuteczne (max|z-y|<0.1): {sum(r['maxerr'] < 0.1 for r in res[key])}/20; "
              f"argmax OK: {[r['argmax_ok'] for r in res[key]]}")
    res["eta_sweep"] = eta_sweep()
    for eta, s in res["eta_sweep"].items():
        print(f"eta={eta}: {s['success']}/20, epoki med={np.median(s['epochs']) if s['epochs'] else None}, "
              f"min={min(s['epochs'], default=None)}, max={max(s['epochs'], default=None)}")
    res["best_nobias_E"] = best_without_bias()
    print("najlepsze E bez biasu (L-BFGS):", res["best_nobias_E"])
    json.dump(res, open("wyniki.json", "w"), indent=1)
