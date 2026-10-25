"""
Plot the WSINDy recovery produced by wsindy_maxwell.

1) Single run  (after:  ./wsindy_maxwell maxwell_vacuum.bin 0.05 )
       python plot-recovery.py                 # reads wsindy_coefs.csv / wsindy_fit.csv
       python plot-recovery.py --prefix run1   # reads run1_coefs.csv / run1_fit.csv
   Makes: recovery_bars.png    recovered vs true coefficient, one panel per equation
          recovery_heatmap.png full coefficient matrix (rows = equations, cols = library terms)
          recovery_fit.png     weak-form fit: (phi, u_t) vs G*w for every window

2) Noise sweep (runs the C++ code for you)
       python plot-recovery.py --sweep 0 0.01 0.05 0.1 0.2 --seeds 3 --bin maxwell_vacuum.bin
   Makes: recovery_noise_sweep.png   coefficient error, spurious terms, missed terms, residual vs noise

Add --show to open windows (laptop only). Output PNGs go to --outdir (default plots/).
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="wsindy")
ap.add_argument("--outdir", default="plots")
ap.add_argument("--show", action="store_true")
ap.add_argument("--sweep", nargs="+", type=float, help="noise ratios to run, e.g. 0 0.01 0.05 0.1")
ap.add_argument("--seeds", type=int, default=3, help="random seeds per noise level in the sweep")
ap.add_argument("--bin", default="maxwell_vacuum.bin")
ap.add_argument("--exe", default="./wsindy_maxwell")
ap.add_argument("--max-deriv", type=int, default=1)
ap.add_argument("--quadratic", type=int, default=0)
args = ap.parse_args()

if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

out = Path(args.outdir)
out.mkdir(exist_ok=True)
TARGETS = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]


def read_coefs(path):
    """-> dict target -> list of (term, coef, truth), plus dict target -> relres."""
    rows, res = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["target"], []).append((r["term"], float(r["coef"]), float(r["truth"])))
            res[r["target"]] = float(r["relres"])
    return rows, res


def score(rows):
    """per-target: max error on true terms (missing term = full error), #spurious, #missed."""
    err, spur, miss = [], 0, 0
    for tg in TARGETS:
        for term, c, tv in rows[tg]:
            if tv != 0:
                err.append(abs(c - tv))
                miss += (c == 0)
            elif c != 0:
                spur += 1
    return (max(err) if err else np.nan), spur, miss


# ------------------------------------------------------------------ noise sweep
if args.sweep:
    sd = out / "sweep_runs"
    sd.mkdir(exist_ok=True)
    levels = list(args.sweep)
    E, S, M, R = ([[] for _ in levels] for _ in range(4))
    for i, nz in enumerate(levels):
        for sed in range(args.seeds if nz > 0 else 1):
            pre = str(sd / f"n{nz:g}_s{sed}")
            cmd = [args.exe, args.bin, str(nz), str(args.max_deriv), str(args.quadratic), pre, str(1000 + sed)]
            print("running:", " ".join(cmd))
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            rows, res = read_coefs(pre + "_coefs.csv")
            e, s, m = score(rows)
            E[i].append(e); S[i].append(s); M[i].append(m); R[i].append(np.mean(list(res.values())))
            Path(pre + "_fit.csv").unlink(missing_ok=True)      # big, not needed in a sweep
    fig, ax = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    xs = np.arange(len(levels))
    for a, data, title in zip(ax, [E, S, M, R],
                              ["max |coef - true| on true terms", "spurious terms (all 6 equations)",
                               "true terms missed (all 6 equations)", "mean relative residual"]):
        for i, d in enumerate(data):
            a.plot([xs[i]] * len(d), d, "o", color="C0", alpha=0.4)
        a.plot(xs, [np.mean(d) for d in data], "-o", color="C0")
        a.set_xticks(xs); a.set_xticklabels([f"{v:g}" for v in levels])
        a.set_xlabel("noise ratio  (std / RMS of each field)"); a.set_title(title, fontsize=10)
        a.grid(alpha=0.3)
    ax[0].set_yscale("log"); ax[3].set_yscale("log")
    fig.suptitle(f"WSINDy noise sweep  (max_deriv={args.max_deriv}, quadratic={args.quadratic})")
    fig.savefig(out / "recovery_noise_sweep.png", dpi=130)
    print("saved", out / "recovery_noise_sweep.png")
    if args.show:
        plt.show()
    sys.exit()

# ------------------------------------------------------------------ single run
cpath, fpath = Path(f"{args.prefix}_coefs.csv"), Path(f"{args.prefix}_fit.csv")
if not cpath.exists():
    sys.exit(f"{cpath} not found. Run ./wsindy_maxwell maxwell_vacuum.bin first (in this folder).")
rows, res = read_coefs(cpath)

# 1. bar charts
fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
for a, tg in zip(axes.flat, TARGETS):
    show = [(t, c, tv) for t, c, tv in rows[tg] if c != 0 or tv != 0]
    x = np.arange(len(show)); wd = 0.38
    a.bar(x - wd / 2, [tv for _, _, tv in show], wd, color="0.65", label="true")
    a.bar(x + wd / 2, [c for _, c, _ in show], wd, color="C0", label="recovered")
    a.set_xticks(x); a.set_xticklabels([t for t, _, _ in show], rotation=30, ha="right")
    a.axhline(0, color="k", lw=0.6)
    a.set_xlim(-1, max(len(show), 3) - 0.0)
    a.set_ylim(-1.35, 1.35)
    a.set_title(f"d{tg}/dt   (rel. residual {res[tg]:.1e})", fontsize=10)
    for xi, (t, c, tv) in zip(x, show):
        if tv == 0:
            a.annotate("spurious", (xi + wd / 2, c), ha="center", va="bottom", fontsize=7, color="C3")
axes[0, 0].legend()
fig.suptitle("WSINDy recovery of vacuum Maxwell: coefficients (units: E/c, x and ct in um)")
fig.savefig(out / "recovery_bars.png", dpi=130)

# 2. heatmap
terms = [t for t, _, _ in rows["Ex"]]
M = np.array([[c for _, c, _ in rows[tg]] for tg in TARGETS])
T = np.array([[tv for _, _, tv in rows[tg]] for tg in TARGETS])
fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(terms)), 4.5), constrained_layout=True)
im = ax.imshow(M, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
ax.set_xticks(range(len(terms))); ax.set_xticklabels(terms, rotation=60, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels([f"d{t}/dt" for t in TARGETS])
for i, j in zip(*np.nonzero(T)):
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=2))
for i, j in zip(*np.nonzero(M)):
    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="coefficient")
ax.set_title("Recovered coefficients (black boxes = true terms)")
fig.savefig(out / "recovery_heatmap.png", dpi=130)

# 3. weak-form fit
if fpath.exists():
    d = np.genfromtxt(fpath, delimiter=",", names=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for a, tg in zip(axes.flat, TARGETS):
        b, gw = d[f"b_{tg}"], d[f"Gw_{tg}"]
        k = slice(None, None, max(1, len(b) // 8000))
        a.plot(b[k], gw[k], ".", ms=2, alpha=0.4)
        lim = np.abs(b).max() * 1.05
        a.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
        a.set_xlabel("(phi, d%s/dt)  from data" % tg); a.set_ylabel("G w  from recovered model")
        a.set_title(f"{tg}: rel. residual {res[tg]:.1e}", fontsize=10)
    fig.suptitle("Weak-form fit, one point per test-function window (on the line = perfect)")
    fig.savefig(out / "recovery_fit.png", dpi=130)
else:
    print(f"({fpath} not found, skipping fit plot)")

print("saved plots to", out.resolve())
for tg in TARGETS:
    print(f"d{tg}/dt =", " ".join(f"{c:+.4f}*{t}" for t, c, _ in rows[tg] if c != 0) or "0")
if args.show:
    plt.show()
