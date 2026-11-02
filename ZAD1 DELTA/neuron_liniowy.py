import json

import numpy as np


def delta_training(X, Z, eta=0.01, K=100, w_range=(-1, 1), rng=None):
    rng = rng or np.random.default_rng()
    M, N = X.shape
    w0 = rng.uniform(w_range[0], w_range[1], N)
    w = w0.copy()
    mse = np.empty(K)
    for k in range(K):
        for mu in range(M):
            y = w @ X[mu]
            e = Z[mu] - y
            w = w + eta * e * X[mu]
        mse[k] = np.mean((Z - X @ w) ** 2)
    return w0, w, mse


def make_data(N, M, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (M, N))
    Z = rng.uniform(-1, 1, M)
    return X, Z


def run_case(N, M, eta, K, repeats, data_seed=42):
    X, Z = make_data(N, M, data_seed)
    X_pinv = np.linalg.pinv(X)
    w_ls = X_pinv @ Z
    P_null = np.eye(N) - X_pinv @ X
    runs = []
    for r in range(repeats):
        w0, w, mse = delta_training(X, Z, eta, K, rng=np.random.default_rng(100 + r))
        w_pred = w_ls + P_null @ w0
        runs.append(dict(
            w0=w0, w=w, mse=mse,
            mse_final=mse[-1],
            max_abs_err=np.max(np.abs(Z - X @ w)),
            dist_ls=np.linalg.norm(w - w_ls),
            dist_pred=np.linalg.norm(w - w_pred),
            norm_w=np.linalg.norm(w),
        ))
    W = np.array([r["w"] for r in runs])
    spread = max(np.linalg.norm(a - b) for a in W for b in W)
    return dict(
        N=N, M=M, X=X, Z=Z, w_ls=w_ls, runs=runs, spread=spread,
        rank=np.linalg.matrix_rank(X), cond=np.linalg.cond(X),
        mse_ls=np.mean((Z - X @ w_ls) ** 2), norm_ls=np.linalg.norm(w_ls),
        null_dim=N - np.linalg.matrix_rank(X),
    )


def eta_sweep(N, M, etas, K):
    X, Z = make_data(N, M)
    w_ls = np.linalg.pinv(X) @ Z
    out = []
    for eta in etas:
        _, w, mse = delta_training(X, Z, eta, K, rng=np.random.default_rng(100))
        out.append(dict(eta=eta, dist_ls=np.linalg.norm(w - w_ls), mse=mse[-1],
                        excess=mse[-1] - np.mean((Z - X @ w_ls) ** 2)))
    return out


if __name__ == "__main__":
    ETA, K, REPEATS = 0.01, 10000, 5
    CASES = [(2, 6), (4, 4), (6, 3)]
    np.set_printoptions(precision=5, suppress=True)
    summary = {}
    for N, M in CASES:
        res = run_case(N, M, ETA, K, REPEATS)
        print(f"\n=== N={N}, M={M} | rank(X)={res['rank']}, cond(X)={res['cond']:.2f}, dim ker X={res['null_dim']}")
        print("w_LS (pinv):", res["w_ls"], "| MSE_LS =", f"{res['mse_ls']:.3e}")
        for i, r in enumerate(res["runs"], 1):
            print(f"  run {i}: w0={r['w0']}\n         w ={r['w']}  MSE={r['mse_final']:.3e}  "
                  f"|w-w_LS|={r['dist_ls']:.2e}  |w-w_pred|={r['dist_pred']:.2e}")
        print(f"  max odleglosc miedzy wagami koncowymi: {res['spread']:.3e}")
        summary[f"{N}_{M}"] = res
    sweep = eta_sweep(2, 6, [0.1, 0.03, 0.01, 0.003, 0.001], 100000)
    print("\nEta sweep (N=2, M=6):")
    for s in sweep:
        print(f"  eta={s['eta']:<6} |w-w_LS|={s['dist_ls']:.3e}  MSE={s['mse']:.6f}  nadwyzka={s['excess']:.3e}")
