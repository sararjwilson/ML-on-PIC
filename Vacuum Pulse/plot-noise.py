"""
Show what the added noise looks like: one field, one snapshot, at several noise ratios, side by side.
The noise is generated exactly as in wsindy_maxwell.cpp: independent Gaussian at every grid point,
std = ratio * RMS of that field over the WHOLE dataset (all snapshots), not just the snapshot shown.

Usage:
    python plot-noise.py                          # Ey, snapshot 50, ratios 0 .25 .5 .75 1
    python plot-noise.py --field Bx --snap 0
    python plot-noise.py --ratios 0 0.05 0.2 0.5  # any list of ratios
    python plot-noise.py maxwell_vacuum.npz --show
Saves plots/noise_comparison.png (and plots/noise_slice.png, a 1D cut through the middle).
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib

ap = argparse.ArgumentParser()
ap.add_argument("npz", nargs="?", default="maxwell_vacuum.npz")
ap.add_argument("--field", default="Ey", choices=["Ex", "Ey", "Ez", "Bx", "By", "Bz"])
ap.add_argument("--snap", type=int, default=50)
ap.add_argument("--ratios", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1.0])
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--outdir", default="plots")
ap.add_argument("--show", action="store_true")
a = ap.parse_args()
if not a.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(a.npz)
u = d[a.field]                                  # (t, x, z)
nt, nx, nz = u.shape
dx = float(d["dx"])
rms = np.sqrt(np.mean(u ** 2))                  # RMS over all space-time, as in the C++ code
clean = u[a.snap]
vmax = np.abs(clean).max()                      # one colour scale for every panel
x0 = -128e-6
ext = [x0 * 1e6, (x0 + nx * dx) * 1e6] * 2
rng = np.random.default_rng(a.seed)
noise = rng.standard_normal(u.shape)            # one noise field, scaled by each ratio (nested noise levels)

out = Path(a.outdir); out.mkdir(exist_ok=True)
fig, ax = plt.subplots(1, len(a.ratios), figsize=(3.3 * len(a.ratios), 3.8), constrained_layout=True)
ax = np.atleast_1d(ax)
for k, r in enumerate(a.ratios):
    noisy = clean + r * rms * noise[a.snap]
    im = ax[k].imshow(noisy.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[k].set_title(f"{100 * r:g}% noise", fontsize=11)
    ax[k].set_xlabel("x (um)")
    if k == 0:
        ax[k].set_ylabel("z (um)")
fig.colorbar(im, ax=ax, shrink=0.85, label=f"{a.field} (T)")
fig.suptitle(f"{a.field}, snapshot {a.snap}: noise std = ratio x RMS of {a.field} over the whole run "
             f"(RMS = {rms:.3f})", fontsize=11)
fig.savefig(out / "noise_comparison.png", dpi=130)

# 1D cut through the middle of the box (z index nz//2), clean vs noisy
fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
xs = (x0 + (np.arange(nx) + 0.5) * dx) * 1e6
for r in a.ratios:
    ax.plot(xs, (clean + r * rms * noise[a.snap])[:, nz // 2], lw=1, label=f"{100 * r:g}%")
ax.plot(xs, clean[:, nz // 2], "k", lw=2, label="clean")
ax.set_xlabel("x (um)"); ax.set_ylabel(a.field); ax.legend(ncol=3, fontsize=8)
ax.set_title(f"{a.field}, snapshot {a.snap}, cut at the middle z")
fig.savefig(out / "noise_slice.png", dpi=130)
print("saved", out / "noise_comparison.png", "and", out / "noise_slice.png")
print(f"RMS of {a.field} over the run = {rms:.4f}; shown snapshot has max |{a.field}| = {vmax:.4f}")
if a.show:
    plt.show()
