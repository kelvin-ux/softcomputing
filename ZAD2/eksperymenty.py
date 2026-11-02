import json
import string
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from font_generator import add_noise, default_font_size, render_letter
from madaline_ocr import Madaline, normalize

HERE = Path(__file__).resolve().parent
FONT = str(HERE / "times.ttf")
W, H, X0, Y0 = 128, 128, 28, 10
LETTERS = string.ascii_lowercase
REQUIRED_NOISE = [0, 10, 30, 50, 90]
SWEEP_NOISE = list(range(0, 101, 10))
REPEATS = 50
OTHER_FONTS = {
    "Liberation Serif": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "DejaVu Serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "Liberation Sans": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
}
SHIFTS = [1, 2, 4, 8]


def run(*args, stdout=None):
    subprocess.run([sys.executable, *map(str, args)], cwd=HERE, check=True, stdout=stdout)


def required_tests():
    """Eksperyment 1: dokładnie tak jak w scenariuszu z punktu 1.5.1, tylko dla 26 liter."""
    for i, letter in enumerate(LETTERS):
        run("font_generator.py", W, H, X0, Y0, "times.ttf", letter, 0, "training_patterns")
        for noise in REQUIRED_NOISE:
            run("font_generator.py", W, H, X0, Y0, "times.ttf", letter, noise, f"test_patterns_{noise}",
                "--seed", 1000 * noise + i)
    summary = {}
    for noise in REQUIRED_NOISE:
        report = HERE / f"raport_szum_{noise}.txt"
        with open(report, "w", encoding="utf-8") as f:
            run("madaline_ocr.py", "training_patterns", f"test_patterns_{noise}", stdout=f)
        rows = []
        for line in report.read_text(encoding="utf-8").splitlines():
            left, right = line.split(" -> ")
            recognized, conf = right.rsplit(", confidence: ", 1)
            rows.append(dict(test=left[7], rec=recognized[7], conf=float(conf)))
        summary[noise] = rows
    with open(HERE / "raport_wzorce_wzorce.txt", "w", encoding="utf-8") as f:
        run("madaline_ocr.py", "training_patterns", "training_patterns", stdout=f)
    return summary


def network(font=FONT, x=X0):
    size = default_font_size(FONT, H, Y0)          # ten sam rozmiar dla wszystkich krojów
    imgs = {l: render_letter(W, H, x, Y0, font, l, size) for l in LETTERS}
    return imgs, Madaline(np.vstack([normalize(imgs[l].ravel().astype(float)) for l in LETTERS]), list(LETTERS))


def noise_sweep(imgs, net):
    """Eksperyment 2: REPEATS losowań szumu dla każdej litery i każdego poziomu."""
    rng = np.random.default_rng(2026)
    Npix = W * H
    n = np.array([imgs[l].sum() for l in LETTERS])
    out = {}
    for noise in SWEEP_NOISE:
        acc, win, corr, wrong, errors = [], [], [], [], Counter()
        for _ in range(REPEATS):
            for i, l in enumerate(LETTERS):
                x = normalize(add_noise(imgs[l], noise, rng).ravel().astype(float))
                y = net.outputs(x)
                k = int(np.argmax(y))
                acc.append(k == i)
                win.append(y[k]); corr.append(y[i]); wrong.append(np.max(np.delete(y, i)))
                if k != i:
                    errors[(l, LETTERS[k])] += 1
        q = noise / 200                                     # prawd. odwrócenia piksela
        theory = np.mean((1 - q) * np.sqrt(n / (n * (1 - q) + (Npix - n) * q)))
        winners = Counter()
        out[noise] = dict(acc=float(np.mean(acc)), win=float(np.mean(win)), corr=float(np.mean(corr)),
                          corr_sd=float(np.std(corr)), wrong=float(np.mean(wrong)), theory=float(theory),
                          errors=errors.most_common(6))
    # co wygrywa przy obrazie całkowicie losowym
    rec = Counter()
    for _ in range(2000):
        x = normalize(add_noise(imgs["a"], 100, rng).ravel().astype(float))
        rec[LETTERS[net.recognize(x)[0]]] += 1
    order = sorted(LETTERS, key=lambda l: -int(imgs[l].sum()))
    return out, dict(pixels={l: int(imgs[l].sum()) for l in LETTERS}, by_size=order, random_winners=rec.most_common(5))


def robustness(imgs, net):
    """Eksperyment 3: litery bez szumu, ale przesunięte albo w innym kroju."""
    res = {}
    for dx in SHIFTS:
        t, _ = network(x=X0 + dx)
        res[f"przesunięcie o {dx} px"] = evaluate(t, net)
    for name, path in OTHER_FONTS.items():
        if not Path(path).exists():                 # np. na Windows - inne ścieżki czcionek
            print(f"Pomijam krój {name}: brak pliku {path}", file=sys.stderr)
            continue
        t, _ = network(font=path)
        res[name] = evaluate(t, net)
    return res


def evaluate(test_imgs, net):
    ok, win, corr, errs = 0, [], [], []
    for i, l in enumerate(LETTERS):
        y = net.outputs(normalize(test_imgs[l].ravel().astype(float)))
        k = int(np.argmax(y))
        ok += k == i
        win.append(y[k]); corr.append(y[i])
        if k != i:
            errs.append(f"{l}->{LETTERS[k]}")
    return dict(acc=ok / len(LETTERS), win=float(np.mean(win)), corr=float(np.mean(corr)), errors=errs)


if __name__ == "__main__":
    summary = required_tests()
    for noise, rows in summary.items():
        ok = sum(r["test"] == r["rec"] for r in rows)
        print(f"szum {noise:>2}%: {ok}/26 poprawnie, średnia pewność {np.mean([r['conf'] for r in rows]):.3f}, "
              f"błędy: {[r['test'] + '->' + r['rec'] for r in rows if r['test'] != r['rec']]}")
    imgs, net = network()
    sweep, extra = noise_sweep(imgs, net)
    for noise, s in sweep.items():
        print(f"{noise:>3}%  acc={s['acc']:.3f} win={s['win']:.3f} corr={s['corr']:.3f}±{s['corr_sd']:.3f} "
              f"wrong={s['wrong']:.3f} teoria={s['theory']:.3f}  {s['errors'][:3]}")
    print(extra)
    rob = robustness(imgs, net)
    for k, v in rob.items():
        print(k, v)
    json.dump(dict(required=summary, sweep=sweep, extra=extra, robustness=rob,
                   font_size=default_font_size(FONT, H, Y0)),
              open(HERE / "wyniki.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
