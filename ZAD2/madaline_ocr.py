import argparse
import os
import sys

import numpy as np
from PIL import Image

DESCRIPTION_FILE = "description.txt"


def read_description(directory):
    """Lista par (nazwa_pliku, etykieta) z pliku description.txt."""
    entries = []
    with open(os.path.join(directory, DESCRIPTION_FILE), encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Błędna linia w {directory}/{DESCRIPTION_FILE}: {line!r}")
            file_name, label = line.split(":", 1)
            entries.append((file_name.strip(), label.strip()))
    return entries


def image_to_vector(path):
    """Obraz -> wektor pikseli (czarny = 1, biały = 0) spłaszczony wiersz po wierszu."""
    pixels = np.asarray(Image.open(path).convert("L")) < 128
    return pixels.astype(np.float64).ravel(), pixels.shape


def normalize(v):
    """Skalowanie wektora do długości 1 (wektor zerowy zostaje bez zmian)."""
    length = np.linalg.norm(v)
    return v / length if length > 0 else v


def load_patterns(directory):
    """Zwraca macierz znormalizowanych wektorów (jeden wiersz na obraz), etykiety i rozmiar obrazów."""
    vectors, labels, shape = [], [], None
    for file_name, label in read_description(directory):
        v, s = image_to_vector(os.path.join(directory, file_name))
        if shape is not None and s != shape:
            raise ValueError(f"{file_name}: rozmiar {s[1]}x{s[0]} różny od pozostałych ({shape[1]}x{shape[0]})")
        shape = s
        vectors.append(normalize(v))
        labels.append(label)
    if not vectors:
        raise ValueError(f"Brak wzorców w katalogu {directory}")
    return np.vstack(vectors), labels, shape


class Madaline:
    """Sieć MADALINE: wiersz l macierzy W to wektor wag neuronu l."""

    def __init__(self, weights, labels):
        self.W = weights
        self.labels = labels

    @classmethod
    def from_directory(cls, directory):
        W, labels, shape = load_patterns(directory)
        net = cls(W, labels)
        net.shape = shape
        return net

    def outputs(self, x):
        """Propagacja: y_l = w^(l) . x dla wszystkich neuronów naraz."""
        return self.W @ x

    def recognize(self, x):
        """Indeks neuronu zwycięskiego i jego wyjście (współczynnik pewności)."""
        y = self.outputs(x)
        winner = int(np.argmax(y))
        return winner, float(y[winner])


def main(argv=None):
    p = argparse.ArgumentParser(description="Rozpoznawanie znaków siecią MADALINE.")
    p.add_argument("train_directory", help="katalog ze wzorcami (png + description.txt)")
    p.add_argument("test_directory", help="katalog ze znakami testowymi (png + description.txt)")
    args = p.parse_args(argv)
    try:
        net = Madaline.from_directory(args.train_directory)
        X, test_labels, shape = load_patterns(args.test_directory)
    except (OSError, ValueError) as e:
        sys.exit(f"Błąd: {e}")
    if shape != net.shape:
        sys.exit(f"Błąd: obrazy testowe {shape[1]}x{shape[0]} mają inny rozmiar niż wzorcowe "
                 f"{net.shape[1]}x{net.shape[0]}")
    for x, test_label in zip(X, test_labels):
        winner, confidence = net.recognize(x)
        print(f"{test_label} -> {net.labels[winner]}, confidence: {confidence:.3f}")


if __name__ == "__main__":
    main()
