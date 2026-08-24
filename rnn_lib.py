import numpy as np
import random

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx] = 1
    return v

def build_vocab(text):
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char

class RNNCell:
    def __init__(self, vocab_size, n_a, rng):
        self.n_a = n_a
        self.Waa = rng.standard_normal((n_a, n_a)) * 0.1
        self.Wax = rng.standard_normal((n_a, vocab_size)) * 0.1
        self.ba = np.zeros((n_a, 1))

    def params(self):
        return {"Waa": self.Waa, "Wax": self.Wax, "ba": self.ba}

    def forward_step(self, x_t, a_prev):
        a_t = np.tanh(self.Waa @ a_prev + self.Wax @ x_t + self.ba)
        cache = (x_t, a_prev, a_t)
        return a_t, cache

    def backward_step(self, da_t, cache):
        x_t, a_prev, a_t = cache
        dz = da_t * (1 - a_t ** 2)
        grads = {
            "Waa": dz @ a_prev.T,
            "Wax": dz @ x_t.T,
            "ba": dz
        }
        da_prev = self.Waa.T @ dz
        return da_prev, grads

class GRUCell:
    def __init__(self, vocab_size, n_a, rng):
        self.n_a = n_a
        concat_size = n_a + vocab_size
        self.Wr = rng.standard_normal((n_a, concat_size)) * 0.1
        self.Wu = rng.standard_normal((n_a, concat_size)) * 0.1
        self.Wc = rng.standard_normal((n_a, concat_size)) * 0.1
        self.br = np.zeros((n_a, 1))
        self.bu = np.zeros((n_a, 1))
        self.bc = np.zeros((n_a, 1))

    def params(self):
        return {
            "Wr": self.Wr,
            "Wu": self.Wu,
            "Wc": self.Wc,
            "br": self.br,
            "bu": self.bu,
            "bc": self.bc
        }

    def forward_step(self, x_t, a_prev):
        concat = np.vstack([a_prev, x_t])
        gr = sigmoid(self.Wr @ concat + self.br)
        gu = sigmoid(self.Wu @ concat + self.bu)
        concat_r = np.vstack([gr * a_prev, x_t])
        a_tilde = np.tanh(self.Wc @ concat_r + self.bc)
        a_t = gu * a_tilde + (1 - gu) * a_prev
        cache = (x_t, a_prev, concat, gr, gu, concat_r, a_tilde, a_t)
        return a_t, cache

    def backward_step(self, da_t, cache):
        x_t, a_prev, concat, gr, gu, concat_r, a_tilde, a_t = cache

        d_a_tilde = da_t * gu
        d_gu = da_t * (a_tilde - a_prev)
        da_prev_direct = da_t * (1 - gu)

        dz_c = d_a_tilde * (1 - a_tilde ** 2)
        dWc = dz_c @ concat_r.T
        dbc = dz_c

        dconcat_r = self.Wc.T @ dz_c
        d_gr_times_aprev = dconcat_r[:self.n_a, :]
        d_gr = d_gr_times_aprev * a_prev
        da_prev_via_gr = d_gr_times_aprev * gr

        dz_u = d_gu * gu * (1 - gu)
        dWu = dz_u @ concat.T
        dbu = dz_u

        dconcat_u = self.Wu.T @ dz_u
        da_prev_via_u = dconcat_u[:self.n_a, :]

        dz_r = d_gr * gr * (1 - gr)
        dWr = dz_r @ concat.T
        dbr = dz_r

        dconcat_r2 = self.Wr.T @ dz_r
        da_prev_via_r = dconcat_r2[:self.n_a, :]

        da_prev = (
            da_prev_direct +
            da_prev_via_gr +
            da_prev_via_u +
            da_prev_via_r
        )

        grads = {
            "Wr": dWr,
            "Wu": dWu,
            "Wc": dWc,
            "br": dbr,
            "bu": dbu,
            "bc": dbc
        }

        return da_prev, grads

CELL_TYPES = {
    "RNN": RNNCell,
    "GRU": GRUCell
}

class CharModelScratch:
    def __init__(self, vocab_size, arch="RNN", n_a=32, seed=1):
        if arch not in CELL_TYPES:
            raise ValueError("arch must be 'RNN' or 'GRU'")

        rng = np.random.default_rng(seed)
        self.n_a = n_a
        self.vocab_size = vocab_size
        self.cell = CELL_TYPES[arch](vocab_size, n_a, rng)
        self.Wya = rng.standard_normal((vocab_size, n_a)) * 0.1
        self.by = np.zeros((vocab_size, 1))

    def train_step(self, x_idx, y_idx, lr=0.1):
        T = len(x_idx)
        a_prev = np.zeros((self.n_a, 1))
        caches = []
        y_hats = []

        for t in range(T):
            x_t = one_hot(x_idx[t], self.vocab_size)
            a_t, cache = self.cell.forward_step(x_t, a_prev)
            y_hat = softmax(self.Wya @ a_t + self.by)
            caches.append(cache)
            y_hats.append(y_hat)
            a_prev = a_t

        loss = sum(
            -np.log(y_hats[t][y_idx[t], 0] + 1e-9)
            for t in range(T)
        )

        cell_grads = {
            k: np.zeros_like(v)
            for k, v in self.cell.params().items()
        }

        dWya = np.zeros_like(self.Wya)
        dby = np.zeros_like(self.by)
        da_next = np.zeros((self.n_a, 1))

        for t in reversed(range(T)):
            dz_y = y_hats[t].copy()
            dz_y[y_idx[t], 0] -= 1

            a_t = caches[t][-1]
            dWya += dz_y @ a_t.T
            dby += dz_y

            da_t = self.Wya.T @ dz_y + da_next
            da_prev, grads = self.cell.backward_step(da_t, caches[t])

            for k in cell_grads:
                cell_grads[k] += grads[k]

            da_next = da_prev

        all_grads = list(cell_grads.values()) + [dWya, dby]
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in all_grads))
        max_norm = 5.0

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for k in cell_grads:
                cell_grads[k] *= scale
            dWya *= scale
            dby *= scale

        for k, v in self.cell.params().items():
            v -= lr * cell_grads[k]

        self.Wya -= lr * dWya
        self.by -= lr * dby

        return loss / T

    def generate(self, seed_idx, num_chars, temperature=0.8, rng=None):
        if len(seed_idx) == 0:
            raise ValueError("seed_idx cannot be empty")

        if rng is None:
            rng = np.random.default_rng()

        a_prev = np.zeros((self.n_a, 1))

        for idx in seed_idx[:-1]:
            x_t = one_hot(idx, self.vocab_size)
            a_prev, _ = self.cell.forward_step(x_t, a_prev)

        cur_idx = seed_idx[-1]
        result = []

        for _ in range(num_chars):
            x_t = one_hot(cur_idx, self.vocab_size)
            a_prev, _ = self.cell.forward_step(x_t, a_prev)
            logits = self.Wya @ a_prev + self.by
            probs = softmax(logits / max(temperature, 1e-6)).ravel()
            probs = probs / probs.sum()
            cur_idx = rng.choice(self.vocab_size, p=probs)
            result.append(cur_idx)

        return result

def train(model, idx_seq, steps=300, seq_len=25, lr=0.1, on_step=None):
    if len(idx_seq) <= seq_len:
        raise ValueError(
            f"Text length ({len(idx_seq)}) must be greater than seq_len ({seq_len})"
        )

    losses = []

    for step in range(steps):
        start = random.randint(0, len(idx_seq) - seq_len - 1)
        x_idx = idx_seq[start:start + seq_len]
        y_idx = idx_seq[start + 1:start + seq_len + 1]
        loss = model.train_step(x_idx, y_idx, lr=lr)
        losses.append(loss)

        if on_step:
            on_step(step, loss)

    return losses