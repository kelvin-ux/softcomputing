# Zadanie 4 – sieć SOM do kompresji obrazów (Wykład 3, Algorytmy 2.1 i 2.2)

Wymagania: Python 3, `pip install numpy pillow matplotlib`

    python som.py images/zelda.png --neurons 64 --out zelda_K64.png   # kompresja jednego obrazu + raport
    python eksperymenty.py    # wszystkie eksperymenty ze sprawozdania -> wyniki.json, odtworzone/ (~3 min)
    python wykresy.py         # fig_*.pdf

Opcje som.py: --frame (rozmiar ramki, domyślnie 2), --epochs (20), --eta (0.2), --q (10), --seed (0),
--weight-bits (8), --init random|data (random = Alg. 2.1).

Sprawozdanie: `sprawozdanie/sprawozdanie.tex` (UTF-8), kompilacja: pdflatex → bibtex → pdflatex ×2.
