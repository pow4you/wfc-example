#!/usr/bin/env python3
"""Fast overlapping Wave Function Collapse.

img -> resize to 200px -> quantize -> extract NxN patterns -> collapse a
300x300 output, starting from a random pattern in the center.

usage: python wfc.py in.png out.png [--colors 8] [--n 3] [--seed 1]
"""
import argparse
import numpy as np
import heapq
import numba as nb
from PIL import Image

DIRS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int64)  # E S W N
OPP = np.array([2, 3, 0, 1], dtype=np.int64)


@nb.njit(cache=True)
def popcount(w):
    c = 0
    while w:
        w &= w - np.uint64(1)
        c += 1
    return c


@nb.njit(cache=True)
def build_compat(pats, n):
    P = pats.shape[0]
    Wd = (P + 63) // 64
    comp = np.zeros((4, P, Wd), dtype=np.uint64)
    for d in range(4):
        dx, dy = DIRS[d, 0], DIRS[d, 1]
        for a in range(P):
            for b in range(P):
                ok = True
                for y in range(n):
                    for x in range(n):
                        yy, xx = y - dy, x - dx
                        if 0 <= yy < n and 0 <= xx < n:
                            if pats[a, y, x] != pats[b, yy, xx]:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    comp[d, a, b >> 6] |= np.uint64(1) << np.uint64(b & 63)
    return comp


@nb.njit(cache=True)
def run(comp, weights, H, W, seed):
    np.random.seed(seed)
    P = weights.shape[0]
    Wd = comp.shape[2]
    N = H * W
    wave = np.zeros((N, Wd), dtype=np.uint64)
    for p in range(P):
        wave[:, p >> 6] |= np.uint64(1) << np.uint64(p & 63)
    counts = np.full(N, P, dtype=np.int64)
    done = np.zeros(N, dtype=np.bool_)
    stack = np.empty(N * 4, dtype=np.int64)
    in_stack = np.zeros(N, dtype=np.bool_)
    allowed = np.zeros(Wd, dtype=np.uint64)

    # heap of (entropy-ish key, cell); lazy invalidation
    heap = [(0.0, 0)]
    heap.pop()

    n_done = 0
    cell = (H // 2) * W + W // 2  # start in the center
    while True:
        # --- collapse `cell` ---
        tot = 0.0
        for p in range(P):
            if (wave[cell, p >> 6] >> np.uint64(p & 63)) & np.uint64(1):
                tot += weights[p]
        r = np.random.random() * tot
        pick = -1
        for p in range(P):
            if (wave[cell, p >> 6] >> np.uint64(p & 63)) & np.uint64(1):
                pick = p
                r -= weights[p]
                if r <= 0:
                    break
        wave[cell, :] = 0
        wave[cell, pick >> 6] = np.uint64(1) << np.uint64(pick & 63)
        counts[cell] = 1
        done[cell] = True
        n_done += 1

        # --- propagate ---
        sp = 0
        stack[sp] = cell
        sp += 1
        in_stack[cell] = True
        while sp > 0:
            sp -= 1
            c = stack[sp]
            in_stack[c] = False
            cx, cy = c % W, c // W
            for d in range(4):
                nx, ny = cx + DIRS[d, 0], cy + DIRS[d, 1]
                if nx < 0 or ny < 0 or nx >= W or ny >= H:
                    continue
                m = ny * W + nx
                if done[m]:
                    # still must stay consistent; check below via AND
                    pass
                allowed[:] = 0
                for wi in range(Wd):
                    w = wave[c, wi]
                    if w == 0:
                        continue
                    for bit in range(64):
                        if (w >> np.uint64(bit)) & np.uint64(1):
                            allowed |= comp[d, wi * 64 + bit]
                changed = False
                newc = 0
                for wi in range(Wd):
                    nw = wave[m, wi] & allowed[wi]
                    if nw != wave[m, wi]:
                        changed = True
                        wave[m, wi] = nw
                    newc += popcount(nw)
                if changed:
                    if newc == 0:
                        return wave, False
                    counts[m] = newc
                    if not in_stack[m]:
                        stack[sp] = m
                        sp += 1
                        in_stack[m] = True
                    if newc == 1 and not done[m]:
                        done[m] = True
                        n_done += 1
                    elif newc > 1:
                        heapq.heappush(heap, (newc + np.random.random() * 0.1, m))
        if n_done >= N:
            return wave, True

        # --- next cell: lowest entropy, lazily validated ---
        cell = -1
        while len(heap) > 0:
            key, m = heapq.heappop(heap)
            if (not done[m]) and counts[m] > 1 and int(key) == counts[m]:
                cell = m
                break
        if cell == -1:  # heap exhausted: scan
            for m in range(N):
                if not done[m] and counts[m] > 1:
                    cell = m
                    break
            if cell == -1:
                return wave, True


def load_quantized(path, size, colors):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    q = im.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
    pal = np.array(q.getpalette()[: colors * 3], dtype=np.uint8).reshape(-1, 3)
    return np.array(q, dtype=np.uint8), pal


def extract_patterns(idx, n, max_patterns):
    h, w = idx.shape
    padded = np.pad(idx, ((0, n - 1), (0, n - 1)), mode="wrap")  # edges count too
    wins = np.stack(
        [padded[y : y + h, x : x + w] for y in range(n) for x in range(n)], axis=-1
    ).reshape(-1, n * n)
    pats, inv, counts = np.unique(wins, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)[:max_patterns]
    remap = np.full(len(pats), -1, dtype=np.int64)
    remap[order] = np.arange(len(order))
    grid = remap[inv.reshape(-1)].reshape(h, w)  # pattern id per source pixel (-1 = dropped)
    return pats[order].reshape(-1, n, n).astype(np.uint8), counts[order].astype(np.float64), grid


def observed_bond_probs(grid, P):
    """prob[d][a, b] = P(b is neighbour of a in direction d | a), from the source image."""
    h, w = grid.shape
    probs = np.zeros((4, P, P))
    for d in range(4):
        dx, dy = DIRS[d]
        ya, yb = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
        xa, xb = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        a, b = grid[ya, xa].ravel(), grid[yb, xb].ravel()
        ok = (a >= 0) & (b >= 0)
        cnt = np.bincount(a[ok] * P + b[ok], minlength=P * P).reshape(P, P).astype(np.float64)
        probs[d] = cnt / np.maximum(cnt.sum(axis=1, keepdims=True), 1)
    return probs


def unpack(comp, P):
    return np.stack(
        [np.unpackbits(comp[d].view(np.uint8), axis=1, bitorder="little")[:, :P].astype(bool) for d in range(4)]
    )


def pack(mask):
    P = mask.shape[1]
    Wd = (P + 63) // 64
    out = np.zeros((4, mask.shape[1], Wd), dtype=np.uint64)
    for d in range(4):
        padded = np.zeros((P, Wd * 64), dtype=bool)
        padded[:, :P] = mask[d]
        out[d] = np.packbits(padded, axis=1, bitorder="little").view(np.uint64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--in-size", type=int, default=200)
    ap.add_argument("--out-size", type=int, default=300)
    ap.add_argument("--colors", type=int, default=8)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-patterns", type=int, default=800)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--preprocess-only", action="store_true",
                    help="only save the resized + quantized image to dst, then exit")
    ap.add_argument("--min-bond", type=float, default=0.0,
                    help="discard bonds (neighbour pairs) whose observed probability in the "
                         "source is below this, e.g. 0.02. 0 = keep every overlap-compatible bond")
    a = ap.parse_args()

    idx, pal = load_quantized(a.src, a.in_size, a.colors)
    print(f"quantized {idx.shape[1]}x{idx.shape[0]}, {len(pal)} colors")
    if a.preprocess_only:
        Image.fromarray(pal[idx]).save(a.dst)
        print("saved", a.dst)
        return

    pats, wts, grid = extract_patterns(idx, a.n, a.max_patterns)
    P = len(pats)
    comp = build_compat(pats, a.n)
    mask = unpack(comp, P)
    if a.min_bond > 0:
        probs = observed_bond_probs(grid, P)
        before = mask.sum()
        for d in range(4):
            # a bond must be likely from both sides (a->b in d, b->a in opposite)
            mask[d] &= (probs[d] >= a.min_bond) & (probs[OPP[d]].T >= a.min_bond)
        print(f"bonds: {before} -> {mask.sum()} (min-bond {a.min_bond})")
    while True:  # drop patterns with no neighbour in some direction (dead ends)
        keep = np.ones(len(pats), dtype=bool)
        for d in range(4):
            keep &= mask[d].any(axis=1) & mask[OPP[d]].any(axis=1)
        if keep.all():
            break
        pats, wts = pats[keep], wts[keep]
        mask = mask[:, keep][:, :, keep]
    if len(pats) == 0:
        raise SystemExit("no patterns left: lower --min-bond or --n")
    print(f"{len(pats)} patterns")
    comp = pack(mask)

    rng = np.random.default_rng(a.seed)
    S = a.out_size
    for t in range(a.tries):
        s = int(rng.integers(0, 2**31 - 1))
        wave, ok = run(comp, wts, S, S, s)
        print(f"try {t}: seed={s} {'ok' if ok else 'contradiction'}")
        if ok:
            break
    P = len(pats)
    out = np.zeros(S * S, dtype=np.uint8)
    for i in range(S * S):
        row = wave[i]
        for p in range(P):
            if (int(row[p >> 6]) >> (p & 63)) & 1:
                out[i] = pats[p, 0, 0]
                break
    Image.fromarray(pal[out.reshape(S, S)]).save(a.dst)
    print("saved", a.dst)


if __name__ == "__main__":
    main()