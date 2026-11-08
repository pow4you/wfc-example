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


def prepare(path, size, colors, n, max_patterns):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    q = im.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
    pal = np.array(q.getpalette()[: colors * 3], dtype=np.uint8).reshape(-1, 3)
    idx = np.array(q, dtype=np.uint8)
    h, w = idx.shape
    # all NxN windows (with wraparound so edges contribute)
    padded = np.pad(idx, ((0, n - 1), (0, n - 1)), mode="wrap")
    wins = np.stack(
        [padded[y : y + h, x : x + w] for y in range(n) for x in range(n)], axis=-1
    ).reshape(-1, n * n)
    pats, counts = np.unique(wins, axis=0, return_counts=True)
    order = np.argsort(-counts)[:max_patterns]
    return pats[order].reshape(-1, n, n).astype(np.uint8), counts[order].astype(np.float64), pal, q.size


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
    a = ap.parse_args()

    pats, wts, pal, qsize = prepare(a.src, a.in_size, a.colors, a.n, a.max_patterns)
    print(f"quantized {qsize}, {len(pal)} colors, {len(pats)} patterns")
    comp = build_compat(pats, a.n)
    while True:  # drop dead-end patterns so truncation can't cause contradictions
        P = len(pats)
        keep = np.ones(P, dtype=bool)
        for d in range(4):
            bits = np.unpackbits(comp[d].view(np.uint8), axis=1, bitorder="little")[:, :P]
            keep &= bits.any(axis=1)
        if keep.all():
            break
        pats, wts = pats[keep], wts[keep]
        comp = build_compat(pats, a.n)
    print(f"{len(pats)} patterns after pruning")
    rng = np.random.default_rng(a.seed)
    S = a.out_size
    for t in range(a.tries):
        s = int(rng.integers(0, 2**31 - 1))
        wave, ok = run(comp, wts, S, S, s)
        print(f"try {t}: seed={s} {'ok' if ok else 'contradiction'}")
        if ok:
            break
    # each cell holds exactly one pattern (or, on failure, take the first bit)
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
