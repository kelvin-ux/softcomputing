import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DESCRIPTION_FILE = "description.txt"


def default_font_size(font_file, h, y):
    """Największy rozmiar czcionki, przy którym pełna wysokość kroju (ascent + descent) mieści się w h - y."""
    ascent, descent = ImageFont.truetype(font_file, 100).getmetrics()
    return max(1, int((h - y) * 100 / (ascent + descent)))


def render_letter(w, h, x, y, font_file, letter, font_size):
    """Rysuje literę w punkcie (x, y) (lewy górny róg wiersza tekstu). Zwraca tablicę (h, w): 1 = czarny, 0 = biały."""
    font = ImageFont.truetype(font_file, font_size)
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), letter, font=font, fill=0)
    left, top, right, bottom = draw.textbbox((x, y), letter, font=font)
    if left < 0 or top < 0 or right > w or bottom > h:
        print(f"Uwaga: litera '{letter}' nie mieści się w obrazie {w}x{h} (ramka {left},{top},{right},{bottom}).",
              file=sys.stderr)
    return (np.asarray(img) < 128).astype(np.uint8)


def add_noise(pixels, noise_level, rng):
    """Każdy piksel z prawdopodobieństwem noise_level/100 otrzymuje losową wartość 0 lub 1.

    0% - obraz bez zmian, 100% - obraz całkowicie losowy (każdy piksel czarny z p = 0,5).
    """
    mask = rng.random(pixels.shape) < noise_level / 100
    noisy = pixels.copy()
    noisy[mask] = rng.integers(0, 2, size=int(mask.sum()))
    return noisy


def save_png(pixels, path):
    """Zapis jako 1-bitowy PNG (czarny = 0, biały = 1 w trybie '1')."""
    Image.fromarray(pixels == 0).save(path)


def update_description(directory, letter, noise_level):
    """Dopisuje linię opisu; jeśli plik <letter>.png był już opisany, podmienia jego linię zamiast dublować."""
    file_name = f"{letter}.png"
    line = f"{file_name}:letter {letter}, noise level {noise_level}%"
    path = os.path.join(directory, DESCRIPTION_FILE)
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
    replaced = False
    for i, old in enumerate(lines):
        if old.split(":", 1)[0].strip() == file_name:
            lines[i], replaced = line, True
    if not replaced:
        lines.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return line


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generator obrazów liter dla sieci MADALINE.")
    p.add_argument("w", type=int, help="szerokość obrazu [px]")
    p.add_argument("h", type=int, help="wysokość obrazu [px]")
    p.add_argument("x", type=int, help="położenie litery - współrzędna x [px]")
    p.add_argument("y", type=int, help="położenie litery - współrzędna y [px]")
    p.add_argument("font_file", help="plik czcionki TrueType (.ttf)")
    p.add_argument("letter", help="pojedynczy znak do narysowania")
    p.add_argument("noise_level", type=int, help="poziom zaszumienia w procentach (0-100)")
    p.add_argument("output_directory", help="katalog wyjściowy")
    p.add_argument("--size", type=int, default=None, help="rozmiar czcionki [px] (domyślnie: dopasowany do h - y)")
    p.add_argument("--seed", type=int, default=None, help="ziarno generatora szumu (powtarzalność)")
    args = p.parse_args(argv)
    if len(args.letter) != 1:
        p.error("parametr letter musi być pojedynczym znakiem")
    if not 0 <= args.noise_level <= 100:
        p.error("noise_level musi należeć do przedziału 0-100")
    if args.w <= 0 or args.h <= 0:
        p.error("w i h muszą być dodatnie")
    return args


def main(argv=None):
    args = parse_args(argv)
    size = args.size or default_font_size(args.font_file, args.h, args.y)
    pixels = render_letter(args.w, args.h, args.x, args.y, args.font_file, args.letter, size)
    pixels = add_noise(pixels, args.noise_level, np.random.default_rng(args.seed))
    os.makedirs(args.output_directory, exist_ok=True)
    save_png(pixels, os.path.join(args.output_directory, f"{args.letter}.png"))
    update_description(args.output_directory, args.letter, args.noise_level)


if __name__ == "__main__":
    main()
