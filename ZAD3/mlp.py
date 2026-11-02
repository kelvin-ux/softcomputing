import numpy as np

X = np.eye(4)          # wzorce wejściowe: [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
Z = np.eye(4)          # wartości pożądane = wejścia (sieć ma kopiować wejście na wyjście)


def f(s):
    """Sigmoidalna funkcja aktywacji."""
    return 1.0 / (1.0 + np.exp(-s))


class MLP:
    def __init__(self, bias=True, w_range=(-1.0, 1.0), rng=None):
        rng = rng or np.random.default_rng()
        self.bias = bias
        self.W1 = rng.uniform(*w_range, size=(2, 4))                       # w_ij^(1)
        self.W2 = rng.uniform(*w_range, size=(4, 3 if bias else 2))        # w_ij^(2), kolumna 3 = waga BIAS

    def hidden(self, x):
        """Krok 4): wyjścia warstwy ukrytej (+ wyjście neuronu obciążającego h_3 = 1)."""
        h = self.W1 @ x
        return np.append(h, 1.0) if self.bias else h

    def forward(self, x):
        """Kroki 4)-6): propagacja sygnału wprzód."""
        h = self.hidden(x)
        s = self.W2 @ h
        return h, f(s)

    def train_pattern(self, x, z, eta):
        """Kroki 4)-9) Algorytmu 2.1 dla jednego wzorca (x, z)."""
        h, y = self.forward(x)
        delta2 = y * (1 - y) * (z - y)                     # 7) f'(s_i)(z_i - y_i), f'(s) = f(s)(1 - f(s))
        delta1 = self.W2[:, :2].T @ delta2                 # 8) sum_k w_ki^(2) delta_k^(2), i = 1, 2
        self.W1 += eta * np.outer(delta1, x)               # 9) w_ij^(1) += eta * delta_i^(1) * x_j
        self.W2 += eta * np.outer(delta2, h)               #    w_ij^(2) += eta * delta_i^(2) * h_j

    def error(self, X=X, Z=Z):
        """Funkcja błędu (2.1): E = 1/2 * sum_k sum_i (z_i - y_i)^2."""
        Y = np.array([self.forward(x)[1] for x in X])
        return 0.5 * np.sum((Z - Y) ** 2)

    def outputs(self, X=X):
        return np.array([self.forward(x)[1] for x in X])

    def hidden_codes(self, X=X):
        return np.array([(self.W1 @ x) for x in X])


def train(bias=True, eta=0.5, epochs=10000, w_range=(-1.0, 1.0), seed=None, shuffle=True, every=1, target=None):
    """Trening z zapisem błędu E co `every` epok (opcjonalnie stop, gdy E < target). Zwraca sieć i historię E."""
    rng = np.random.default_rng(seed)
    net = MLP(bias, w_range, rng)
    history = [net.error()]
    order = np.arange(len(X))
    for epoch in range(1, epochs + 1):                     # 2) pętla po epokach
        if shuffle:
            rng.shuffle(order)
        for k in order:                                    # 3) kolejne wzorce treningowe
            net.train_pattern(X[k], Z[k], eta)
        if epoch % every == 0:
            history.append(net.error())
            if target is not None and history[-1] < target:
                break
    return net, np.array(history)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    for bias in (True, False):
        net, hist = train(bias=bias, eta=0.5, epochs=10000, seed=1)
        print(f"\n=== {'Z BIASEM' if bias else 'BEZ BIASU'}: E po treningu = {hist[-1]:.6f}")
        print("wyjścia sieci (wiersz = wzorzec):\n", net.outputs())
        print("reprezentacja w warstwie ukrytej (h1, h2):\n", net.hidden_codes())
