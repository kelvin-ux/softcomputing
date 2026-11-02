# Zadanie 2 – sieć MADALINE do rozpoznawania znaków

Wymagania: Python 3, `pip install numpy pillow matplotlib`

## Programy (zgodnie z punktem 1.5)

    python font_generator.py 128 128 28 10 times.ttf a 0 training_patterns
    python font_generator.py 128 128 28 10 times.ttf a 30 test_patterns_30
    python madaline_ocr.py training_patterns test_patterns_30

Opcjonalnie: `--size S` (rozmiar czcionki, domyślnie dopasowany do h - y), `--seed S` (powtarzalny szum).

## Odtworzenie wszystkich wyników ze sprawozdania

    python eksperymenty.py    # obrazy, raport_szum_*.txt, wyniki.json
    python wykresy.py         # fig_*.pdf

Sprawozdanie: `sprawozdanie/sprawozdanie.tex` (kodowanie cp1250, jak we wzorze),
kompilacja: pdflatex → bibtex → pdflatex ×2.
