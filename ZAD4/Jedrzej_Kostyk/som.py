"""Kompresja obrazów w skali szarości siecią samoorganizującą się SOM (Wykład 3, Algorytmy 2.1 i 2.2).

Użycie:
    python som.py images/zelda.png --neurons 64 [--frame 2] [--epochs 20] [--eta 0.2] [--q 10]
                  [--seed 0] [--weight-bits 8] [--out odtworzony.png]

Kroki Algorytmu 2.2:
  1-3) obraz N x N dzielony na ramki n x n, każda ramka spłaszczana do wektora n^2 i normalizowana,
  4-5) trening sieci SOM (K neuronów, n^2 wejść) metodą "zwycięzca bierze wszystko" (Alg. 2.1),
  6)   usunięcie neuronów martwych (takich, których wagi ani razu nie zmieniono),
  7-9) kodowanie: lista L (indeks neuronu zwycięskiego dla każdej ramki), lista T (długość ramki / q,
       zaokrąglona), lista W (wagi neuronów),
  10-11) dekodowanie: ramka = q * T_k * W[L_k],
  12)  współczynnik kompresji CR (wzór 2.4) i jakość PSNR (wzór 2.3).
"""
import argparse

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------- obraz <-> ramki
def load_image(path):
    """Obraz w 8-bitowej skali szarości jako tablica float (N x N)."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def image_to_frames(img, n):
    """Krok 2: podział na ramki n x n (wiersz po wierszu) i spłaszczenie każdej do wektora n^2."""
    H, W = img.shape
    if H % n or W % n:
        raise ValueError(f"rozmiar obrazu {W}x{H} nie jest wielokrotnością ramki {n}x{n}")
    return img.reshape(H // n, n, W // n, n).swapaxes(1, 2).reshape(-1, n * n)


def frames_to_image(frames, n, shape):
    """Krok 10: zwinięcie wektorów z powrotem w ramki n x n i złożenie obrazu."""
    H, W = shape
    return frames.reshape(H // n, W // n, n, n).swapaxes(1, 2).reshape(H, W)


def normalize_rows(X):
    """Wzór (2.2): x' = x / ||x||; wiersze zerowe (czarne ramki) zostają zerowe."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=np.zeros_like(X), where=norms > 0), norms.ravel()


# ----------------------------------------------------------------------------- sieć SOM (Alg. 2.1)
class SOM:
    def __init__(self, num_neurons, dim, rng, init_vectors=None):
        if init_vectors is None:
            W = rng.uniform(-1.0, 1.0, size=(num_neurons, dim))        # 1) losowe wagi z [-1, 1]
        else:                                                           # (wariant dodatkowy: losowe ramki obrazu)
            W = init_vectors[rng.choice(len(init_vectors), num_neurons, replace=False)].copy()
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)           #    i ich normalizacja
        self.updated = np.zeros(num_neurons, dtype=bool)                # do wykrycia neuronów martwych

    def outputs(self, x):
        """5) y_i = sum_j w_ij x_j  - wyjścia wszystkich neuronów (iloczyny skalarne)."""
        return self.W @ x

    def train(self, X, epochs, eta, rng, shuffle=True):
        """Kroki 2)-7) Alg. 2.1. X - znormalizowane wektory treningowe (wiersze)."""
        X = X[np.linalg.norm(X, axis=1) > 0]        # ramka czarna nie ma kierunku - pomijamy w treningu
        order = np.arange(len(X))
        for _ in range(epochs):                     # 2) epoki
            if shuffle:
                rng.shuffle(order)
            for k in order:                         # 3) kolejne wzorce
                x = X[k]                            # 4) wzorzec na wejście
                m = int(np.argmax(self.W @ x))      # 5)-6) neuron zwycięski
                w = self.W[m] + eta * (x - self.W[m])           # 6) w_m = w_m + eta (x - w_m)
                self.W[m] = w / np.linalg.norm(w)               # 7) normalizacja w_m
                self.updated[m] = True

    def remove_dead(self):
        """Krok 6) Alg. 2.2: usuwa neurony, których wagi nie zmieniły się ani razu."""
        self.W = self.W[self.updated]
        self.updated = self.updated[self.updated]
        return len(self.W)

    def winners(self, X):
        """Indeks neuronu zwycięskiego dla każdego wiersza X."""
        return np.argmax(X @ self.W.T, axis=1)


# ----------------------------------------------------------------------------- kodowanie / dekodowanie
def bits_for(max_value):
    """floor(log2(v)) + 1 bitów na liczbę całkowitą z przedziału [0, v] (min. 1 bit)."""
    return max(1, int(max_value).bit_length())


def quantize_weights(W, bits):
    """Wagi zapisane na `bits` bitach (równomiernie w [-1, 1]) - tak jak zakłada wzór (2.4)."""
    levels = 2 ** bits - 1
    return np.round((W + 1) / 2 * levels) / levels * 2 - 1


def compress(img, frame=2, neurons=64, epochs=20, eta=0.2, q=10, seed=0, weight_bits=8, shuffle=True,
             init="random"):
    """Pełny Algorytm 2.2. Zwraca słownik z listami L, T, W, obrazem odtworzonym i statystykami."""
    rng = np.random.default_rng(seed)
    frames = image_to_frames(img, frame)                              # 2)
    Xn, lengths = normalize_rows(frames)                              # 3)
    init_vectors = Xn[lengths > 0] if init == "data" else None       # domyślnie: Alg. 2.1, losowo z [-1, 1]
    som = SOM(neurons, frame * frame, rng, init_vectors)              # 4)
    som.train(Xn, epochs, eta, rng, shuffle)                          # 5)
    K = som.remove_dead()                                             # 6)
    L = som.winners(Xn)                                               # 7) indeksy zwycięzców
    T = np.rint(lengths / q).astype(int)                              # 7) skwantowane długości ramek
    W = quantize_weights(som.W, weight_bits)                          # 8) wagi (zapisane na weight_bits)
    decoded = frames_to_image(q * T[:, None] * W[L], frame, img.shape)              # 10)-11)
    decoded = np.clip(np.rint(decoded), 0, 255)                                      # obraz 8-bitowy

    N2 = img.size
    n2 = frame * frame
    bits_index, bits_length = bits_for(K - 1), bits_for(T.max())
    org_bits = 8 * N2
    cmp_bits = weight_bits * K * n2 + len(L) * bits_index + len(T) * bits_length    # mianownik (2.4)
    mse = np.mean((img - decoded) ** 2)
    return dict(
        L=L, T=T, W=W, decoded=decoded,
        resolution=img.shape, frame=frame, num_frames=len(L), neurons_initial=neurons, neurons=K,
        weights_per_neuron=n2, max_quantized_length=int(T.max()),
        bits_per_index=bits_index, bits_per_length=bits_length, bits_per_weight=weight_bits,
        org_size_bits=org_bits, cmp_size_bits=cmp_bits,
        CR=org_bits / cmp_bits, MSE=mse, PSNR=10 * np.log10(255 ** 2 / mse) if mse > 0 else np.inf,
    )


def report(r, show_weights=True):
    lines = [
        f"resolution             = {r['resolution'][1]} x {r['resolution'][0]}",
        f"frame_size             = {r['frame']} x {r['frame']}",
        "",
        f"num_frames             = {r['num_frames']}",
        f"num_neurons (start)    = {r['neurons_initial']}",
        f"num_neurons            = {r['neurons']}",
        f"num_weights_per_neuron = {r['weights_per_neuron']}",
        f"max_quantized_length   = {r['max_quantized_length']}",
        "",
        f"bits_per_index         = {r['bits_per_index']}",
        f"bits_per_length        = {r['bits_per_length']}",
        f"bits_per_weight        = {r['bits_per_weight']}",
        "",
        f"org_size_bits          = {r['org_size_bits']}",
        f"cmp_size_bits          = {r['cmp_size_bits']}",
        "",
    ]
    if show_weights:
        lines += [f"w_{i + 1:<3}= [{'  '.join(f'{v:6.3f}' for v in w)}]" for i, w in enumerate(r["W"][:16])]
        if len(r["W"]) > 16:
            lines.append(f"... (razem {len(r['W'])} neuronów)")
        lines.append("")
    lines += [f"CR   = {r['CR']:.3f}", f"PSNR = {r['PSNR']:.3f} dB"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Kompresja obrazu siecią SOM (Alg. 2.2, Wykład 3).")
    p.add_argument("image", help="obraz w skali szarości (np. images/zelda.png)")
    p.add_argument("--frame", type=int, default=2, help="rozmiar ramki n (ramka n x n), domyślnie 2")
    p.add_argument("--neurons", type=int, default=64, help="początkowa liczba neuronów K")
    p.add_argument("--epochs", type=int, default=20, help="liczba epok treningu")
    p.add_argument("--eta", type=float, default=0.2, help="współczynnik nauczania")
    p.add_argument("--q", type=int, default=10, help="współczynnik kwantyzacji długości ramek")
    p.add_argument("--seed", type=int, default=0, help="ziarno generatora (wagi początkowe, kolejność)")
    p.add_argument("--weight-bits", type=int, default=8, help="bity na jedną wagę w pliku (wzór 2.4: 8)")
    p.add_argument("--init", choices=["random", "data"], default="random",
                   help="inicjalizacja wag: random (Alg. 2.1) lub data (losowe ramki obrazu - wariant dodatkowy)")
    p.add_argument("--out", help="zapisz obraz odtworzony do pliku png")
    a = p.parse_args(argv)
    r = compress(load_image(a.image), a.frame, a.neurons, a.epochs, a.eta, a.q, a.seed, a.weight_bits,
                 init=a.init)
    print(report(r))
    if a.out:
        Image.fromarray(r["decoded"].astype(np.uint8)).save(a.out)


if __name__ == "__main__":
    main()
