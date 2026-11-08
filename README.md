# wfc.py

A fast overlapping **Wave Function Collapse** image generator.

Takes any RGB image, shrinks it to 200 px, quantizes the colors, learns which
small patterns sit next to which, and then collapses a new 300×300 image
starting from a random pattern in the center.

## Install

```bash
pip install numpy pillow numba
```

The first run compiles the numba kernels, so expect a short delay once.

## Usage

```bash
python wfc.py input.png output.png
```

Look at the preprocessing result only (no WFC):

```bash
python wfc.py input.png pre.png --preprocess-only --colors 8
```

A more forgiving run:

```bash
python wfc.py input.png output.png --colors 6 --drop-bonds 30 --error-allowance 0.03 --seed 42
```

## How it works

1. **Preprocess** – resize to fit 200 px with nearest-neighbor (sharp, MS Paint style; change with
   `--resample`), then quantize to `--colors` (median cut, no dithering).
2. **Patterns** – collect every `N×N` window (wrapping at the edges), dedupe, keep the
   `--max-patterns` most frequent.
3. **Bonds** – two patterns may be neighbors if their overlap matches.
   With `--drop-bonds`, each pattern's neighbors are ranked by how often they appear
   next to it in the source, and the lowest percentage is thrown out.
   Patterns with no neighbor left in some direction are pruned.
4. **Collapse** – start in the center cell, repeatedly collapse the lowest-entropy cell
   (weighted by pattern frequency), and propagate constraints with bitsets.
5. **Errors** – if a cell would end up with no valid pattern, the constraint is ignored and
   counted as a violation. A try is abandoned only when violations exceed
   `--error-allowance`; then it retries with a new seed.
6. **Output** – each cell becomes the top-left pixel of its pattern, mapped back to the palette.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--in-size` | 200 | Longest side of the preprocessed source |
| `--out-size` | 300 | Output width and height |
| `--colors` | 8 | Palette size after quantization |
| `--resample` | nearest | Shrink method: `nearest` (sharp), `box`, `bilinear`, `lanczos` (these average and soften) |
| `--n` | 2 | Pattern size (`N×N`). Larger means more structure, slower, more contradictions |
| `--max-patterns` | 800 | Keep only the most frequent patterns |
| `--drop-bonds PCT` | 0 | Per pattern and direction, discard the lowest PCT % of bonds (0–99) |
| `--error-allowance FRAC` | 0.02 | Fraction of cells allowed to violate a constraint |
| `--preprocess-only` | off | Save the resized and quantized image, then exit |
| `--seed` | random | Seed for reproducible runs |
| `--tries` | 50 | Max retries after a failed attempt |

## Tuning tips

- **Too many failed tries** – raise `--error-allowance`, lower `--colors`, lower `--n`,
  or lower `--max-patterns`.
- **Too noisy** – lower `--error-allowance`, or try `--n 3` with few colors and few patterns.
- **Too much like the source** – lower `--drop-bonds`.
- **Too slow** – fewer colors and fewer patterns. Cost grows quickly with `--n` and pattern count.
- Noisy or photographic inputs produce lots of unique patterns. Smooth, graphic images
  with few colors work best.