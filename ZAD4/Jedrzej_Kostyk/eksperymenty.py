"""Eksperymenty do sprawozdania z zadania 4 (SOM, kompresja obrazów). Wyniki -> wyniki.json, obrazy -> odtworzone/."""
import json
import os

import numpy as np
from PIL import Image

from som import compress, load_image, normalize_rows, image_to_frames

IMAGES = ["zelda", "cameraman", "peppers"]
NEURONS = [4, 8, 16, 32, 64, 128, 256]
SEEDS = range(5)
BASE = dict(frame=2, epochs=20, eta=0.2, q=10)          # parametry sugerowane w specyfikacji


def stats(img, **kw):  # noqa: D401
    """Średnie i odchylenia CR, PSNR i liczby żywych neuronów po SEEDS uruchomieniach."""
    rs = [compress(img, seed=s, **kw) for s in SEEDS]
    out = {}
    for key in ("CR", "PSNR", "neurons", "bits_per_index", "bits_per_length", "cmp_size_bits"):
        v = np.array([r[key] for r in rs], dtype=float)
        out[key] = float(v.mean())
        out[key + "_sd"] = float(v.std())
    return out


def param_selection(img):
    """Punkt 3 specyfikacji: wpływ eta, liczby epok i q (obraz zelda, K = 64)."""
    res = {"eta": {}, "epochs": {}, "q": {}}
    for eta in [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]:
        res["eta"][str(eta)] = stats(img, neurons=64, **{**BASE, "eta": eta})
    for ep in [1, 2, 5, 10, 20, 50]:
        res["epochs"][str(ep)] = stats(img, neurons=64, **{**BASE, "epochs": ep})
    for q in [1, 2, 5, 10, 20, 40]:
        res["q"][str(q)] = stats(img, neurons=64, **{**BASE, "q": q})
    return res


def main_sweep(images):
    """Punkt 4 specyfikacji: K = 4..256, ramka 2x2, dla każdego obrazu."""
    return {name: {str(K): stats(img, neurons=K, **BASE) for K in NEURONS} for name, img in images.items()}


def frame_sweep(img):
    """Punkt 5 (opcjonalny): inne rozmiary ramek."""
    return {str(n): {str(K): stats(img, neurons=K, **{**BASE, "frame": n}) for K in NEURONS} for n in (2, 4, 8)}


def geometry(img):
    """Jak blisko siebie leżą znormalizowane ramki: kosinus każdej ramki z ramką ,,płaską'' [1,...,1]/n."""
    out = {}
    for n in (2, 4, 8):
        Xn, lengths = normalize_rows(image_to_frames(img, n))
        flat = np.ones(n * n) / n
        c = Xn[lengths > 0] @ flat
        rng = np.random.default_rng(0)
        R = rng.uniform(-1, 1, (100000, n * n)); R /= np.linalg.norm(R, axis=1, keepdims=True)
        out[str(n)] = dict(cos_min=float(c.min()), cos_median=float(np.median(c)),
                           random_cos_median=float(np.median(R @ flat)),
                           random_cos_above_min=float(np.mean(R @ flat > c.min())))
    return out


def data_init(img):
    """Eksperyment dodatkowy: wagi startowe = losowe ramki obrazu zamiast losowych liczb z [-1, 1]."""
    return {str(n): {str(K): stats(img, neurons=K, **{**BASE, "frame": n}, init="data") for K in NEURONS}
            for n in (2, 4)}


def save_decoded(images):
    os.makedirs("odtworzone", exist_ok=True)
    for name, img in images.items():
        for K in (4, 16, 256):
            r = compress(img, neurons=K, seed=0, **BASE)
            Image.fromarray(r["decoded"].astype(np.uint8)).save(f"odtworzone/{name}_n2_K{K}.png")
    for n in (4, 8):
        for K in (16, 256):
            r = compress(images["zelda"], neurons=K, seed=0, **{**BASE, "frame": n})
            Image.fromarray(r["decoded"].astype(np.uint8)).save(f"odtworzone/zelda_n{n}_K{K}.png")
    for n, K in ((2, 256), (4, 256)):
        r = compress(images["zelda"], neurons=K, seed=0, **{**BASE, "frame": n}, init="data")
        Image.fromarray(r["decoded"].astype(np.uint8)).save(f"odtworzone/zelda_n{n}_K{K}_data.png")


if __name__ == "__main__":
    images = {name: load_image(f"images/{name}.png") for name in IMAGES}
    out = dict(base=BASE, neurons=NEURONS, seeds=len(SEEDS))
    out["params"] = param_selection(images["zelda"])
    for k, d in out["params"].items():
        print(k, {v: (round(s["PSNR"], 2), round(s["CR"], 3), round(s["neurons"], 1)) for v, s in d.items()})
    out["main"] = main_sweep(images)
    for name, d in out["main"].items():
        print(name, {K: (round(s["CR"], 3), round(s["PSNR"], 2), round(s["neurons"], 1)) for K, s in d.items()})
    out["frames"] = frame_sweep(images["zelda"])
    for n, d in out["frames"].items():
        print("frame", n, {K: (round(s["CR"], 2), round(s["PSNR"], 2), round(s["neurons"], 1)) for K, s in d.items()})
    out["geometry"] = geometry(images["zelda"])
    print(out["geometry"])
    out["data_init"] = data_init(images["zelda"])
    for n, d in out["data_init"].items():
        print("data-init frame", n, {K: (round(s["CR"], 2), round(s["PSNR"], 2), round(s["neurons"], 1)) for K, s in d.items()})
    save_decoded(images)
    json.dump(out, open("wyniki.json", "w"), indent=1)
