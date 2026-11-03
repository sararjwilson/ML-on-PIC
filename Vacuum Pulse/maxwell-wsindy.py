"""
WSINDy recovery of 2D vacuum Maxwell from a WarpX run -- everything in one script.

    conda activate warpx
    python maxwell-wsindy.py run                  # run WarpX (imputs-vacuumpulse.txt) -> run/diags/diag1
    python maxwell-wsindy.py convert              # openPMD output -> maxwell_vacuum.npz (E/c, t -> c*t)
    python maxwell-wsindy.py fit --noise 0.05     # WSINDy -> wsindy_coefs.csv, wsindy_fit.csv
    python maxwell-wsindy.py plot                 # recovery plots + noise plots from those csv files
    python maxwell-wsindy.py sweep --sweep 0 0.01 0.05 0.1 0.2 --seeds 3
    python maxwell-wsindy.py all --noise 0.05     # everything above (simulation skipped if the .npz exists;
                                                  #   add --rerun to force it, --sweep ... to add a sweep)

Plots go to --outdir (default plots/): one PNG per figure, and every figure from the same command
is also collected into plots/all_plots.pdf.  Add --show to open windows.

Common options: --npz FILE  --prefix NAME  --seed N  --max-deriv K  --quadratic 0/1
Noise plot options: --field Ey --snap 50 --ratios 0 0.25 0.5 0.75 1 --x0 -128 (left edge of box, um)

Noise model (fit, sweep and noise plot all use the same one): independent Gaussian at every grid
point, std = ratio * RMS of that field over the WHOLE dataset; field f uses generator seed (seed, f),
so the noise plot shows exactly the noise the fit sees for the same --seed.
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
FN = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]

# expected vacuum Maxwell (units: E/c, x and ct in the same length unit)
TRUTH = {
    "Ex": {"By_z": -1},
    "Ey": {"Bx_z": 1, "Bz_x": -1},
    "Ez": {"By_x": 1},
    "Bx": {"Ey_z": 1},
    "By": {"Ez_x": 1, "Ex_z": -1},
    "Bz": {"Ey_x": -1},
}

# WSINDy settings
P, Q = 4, 4                       # test fn (s-a)^p (b-s)^q ; need p,q >= 3 for 2nd derivatives
WIN = (21, 25, 25)                # window length (grid points) in t, x, z
STRIDE = (3, 4, 4)                # stride between window starts in t, x, z


# =============================================================== 1. simulation
def cmd_run(a):
    inputs = HERE / "imputs-vacuumpulse.txt"      # your input file (note the filename spelling)
    rundir = HERE / "run"
    if not inputs.exists():
        sys.exit(f"Input file not found: {inputs}")
    exe = shutil.which("warpx.2d") or shutil.which("warpx")
    if exe is None:
        sys.exit("No WarpX executable found. Activate the env: conda activate warpx")
    if rundir.exists():                           # start clean so stale snapshots don't get mixed in
        shutil.rmtree(rundir)
    rundir.mkdir()
    print("Using:", exe)
    subprocess.run([exe, str(inputs)], cwd=rundir, check=True)
    diag = rundir / "diags" / "diag1"
    print("Done. Snapshots in:", diag)
    print(sorted(p.name for p in diag.iterdir())[:5], "...")


# =============================================================== 2. openPMD -> npz
def cmd_convert(a):
    from openpmd_viewer import OpenPMDTimeSeries
    from scipy.constants import c

    ts = OpenPMDTimeSeries(str(HERE / "run" / "diags" / "diag1"))
    its = ts.iterations
    print("snapshots:", len(its))

    def load(field, coord):
        out = []
        for it in its:
            F, info = ts.get_field(field=field, coord=coord, iteration=it)
            axes = [info.axes[i] for i in range(F.ndim)]
            if axes == ["z", "x"]:
                F = F.T                           # force (x, z) ordering
            elif axes != ["x", "z"]:
                raise ValueError(f"Unexpected axes {axes}; expected 2D x/z")
            out.append(F)
        return np.stack(out), info

    data = {}
    for n in ["Ex", "Ey", "Ez"]:
        data[n], info = load("E", n[1])
        data[n] = data[n] / c
    for n in ["Bx", "By", "Bz"]:
        data[n], info = load("B", n[1])
    print("shape (t, x, z):", data["Ex"].shape)
    print({n: float(np.abs(v).max()) for n, v in data.items()})
    np.savez(a.npz, t=c * ts.t, dx=info.x[1] - info.x[0], dz=info.z[1] - info.z[0], **data)
    print("saved", a.npz)


def load_data(path):
    """-> F (6, nt, nx, nz) and (dt, dx, dz) in microns. Reads the .npz (metres) or an old .bin (microns)."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"{path} not found. Make it with:  python {Path(__file__).name} run  then  convert")
    if path.suffix == ".bin":
        with open(path, "rb") as fh:
            nt, nx, nz = (int(v) for v in np.fromfile(fh, np.int32, 3))
            h = np.fromfile(fh, np.float64, 3)
            F = np.fromfile(fh, np.float64, 6 * nt * nx * nz).reshape(6, nt, nx, nz)
        return F, tuple(float(v) for v in h)
    d = np.load(path)
    t = np.asarray(d["t"], float)
    steps = np.diff(t)
    if np.ptp(steps) > 1e-6 * abs(steps.mean()):
        print("warning: snapshots are not evenly spaced in time; using the mean spacing")
    F = np.stack([d[n] for n in FN]).astype(float)
    return F, (steps.mean() * 1e6, float(d["dx"]) * 1e6, float(d["dz"]) * 1e6)


def noise_field(shape, ratio, rms, seed, f):
    """Gaussian noise for field f: std = ratio * rms. Same draw for every ratio (nested noise levels)."""
    rng = np.random.default_rng([seed, f, int(round(ratio * 1e6))])
    return ratio * rms * rng.standard_normal(shape)

def add_noise(F, ratio, seed):
    if ratio <= 0:
        return F
    return np.stack([F[f] + noise_field(F[f].shape, ratio, np.sqrt(np.mean(F[f] ** 2)), seed, f)
                     for f in range(6)])


# =============================================================== 3. WSINDy
class Axis:
    def __init__(self, n, h, m, stride):
        if n < m:
            sys.exit(f"axis has {n} points, fewer than the window length {m}")
        self.m, self.stride = m, stride
        self.starts = np.arange(0, n - m + 1, stride)        # first grid index of each window
        b = (m - 1) * h
        C = 1.0 / (P ** P * Q ** Q) * ((P + Q) / b) ** (P + Q)
        f = np.arange(m) * h
        g = b - f
        self.d = [                                           # d[k][i] = k-th derivative of phi at point i
            C * f ** P * g ** Q,
            C * (P * f ** (P - 1) * g ** Q - Q * f ** P * g ** (Q - 1)),
            C * (P * (P - 1) * f ** (P - 2) * g ** Q - 2.0 * P * Q * f ** (P - 1) * g ** (Q - 1)
                 + Q * (Q - 1) * f ** P * g ** (Q - 2)),
        ]


class WeakForm:
    def __init__(self, shape, h):
        self.h = h
        self.ax = [Axis(n, hh, m, s) for n, hh, m, s in zip(shape, h, WIN, STRIDE)]
        self.nrows = int(np.prod([A.starts.size for A in self.ax]))

    def column(self, data, ot, ox, oz):
        """Every window: (-1)^(ot+ox+oz) * integral of (d^ot_t d^ox_x d^oz_z phi) * data.
        phi is separable -> one axis at a time. Row order: (jt*Nx + jx)*Nz + jz."""
        AT, AX, AZ = self.ax
        A = sliding_window_view(data, AZ.m, axis=2)[:, :, ::AZ.stride] @ AZ.d[oz]
        B = sliding_window_view(A, AX.m, axis=1)[:, ::AX.stride] @ AX.d[ox]
        out = sliding_window_view(B, AT.m, axis=0)[::AT.stride] @ AT.d[ot]
        sign = -1.0 if (ot + ox + oz) % 2 else 1.0
        return sign * np.prod(self.h) * out.ravel()


def build_terms(max_deriv, quadratic):
    """(name, f1, f2, bx, bz); f2 = -1 for linear terms."""
    T = []
    for f in range(6):
        for bx in range(max_deriv + 1):
            for bz in range(max_deriv - bx + 1):
                nm = FN[f] + ("_" + "x" * bx + "z" * bz if bx + bz else "")
                T.append((nm, f, -1, bx, bz))
    if quadratic:
        T += [(FN[f1] + "*" + FN[f2], f1, f2, 0, 0) for f1 in range(6) for f2 in range(f1, 6)]
    return T


def lsq(A, b):        # minimum-norm least squares: rank-safe (div B = 0, div E = 0 make columns collinear)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def mstls(Theta, target, lam):
    with np.errstate(divide="ignore"):
        r = np.linalg.norm(target) / np.linalg.norm(Theta, axis=0)
    L, U = lam * np.maximum(1.0, r), np.minimum(1.0, r) / lam
    idx = np.arange(Theta.shape[1])
    while idx.size:
        prior = idx.size
        xi = np.abs(lsq(Theta[:, idx], target))
        idx = idx[(xi >= L[idx]) & (xi <= U[idx])]
        if idx.size == prior:
            break
    w = np.zeros(Theta.shape[1])
    if idx.size:
        w[idx] = lsq(Theta[:, idx], target)
    return w


def mstls_scan(G, b):
    """Choose lambda by scanning (Messenger & Bortz 2021): distance to the full least-squares fit,
    measured through the library, plus a sparsity penalty."""
    wstar = lsq(G, b)
    denom = np.linalg.norm(G @ wstar)
    best, best_lam, best_loss = np.zeros(G.shape[1]), 0.0, np.inf
    for k in range(60):
        lam = 10.0 ** (-4.0 + 4.0 * k / 59.0 * 0.95)        # 1e-4 ... ~0.6
        w = mstls(G, b, lam)
        nnz = np.count_nonzero(w)
        if nnz == 0:
            continue
        loss = np.linalg.norm(G @ (w - wstar)) / denom + nnz / G.shape[1]
        if loss < best_loss:
            best_loss, best, best_lam = loss, w, lam
    return best, best_lam


def wsindy(F, h, noise, seed, max_deriv, quadratic, verbose=True):
    F = add_noise(F, noise, seed)
    wf = WeakForm(F.shape[1:], h)
    terms = build_terms(max_deriv, quadratic)
    if verbose:
        print("windows: " + " x ".join(str(A.starts.size) for A in wf.ax) +
              f" = {wf.nrows} rows,  library terms: {len(terms)},  noise ratio {noise}")
    G = np.column_stack([wf.column(F[f1], 0, bx, bz) if f2 < 0 else wf.column(F[f1] * F[f2], 0, 0, 0)
                         for _, f1, f2, bx, bz in terms])
    R = dict(terms=[t[0] for t in terms], W=[], B=[], GW=[], lam=[], res=[])
    for f in range(6):
        b = wf.column(F[f], 1, 0, 0)                         # -(phi_t, u) = (phi, u_t)
        w, lam = mstls_scan(G, b)
        R["W"].append(w); R["B"].append(b); R["GW"].append(G @ w); R["lam"].append(lam)
        R["res"].append(np.linalg.norm(G @ w - b) / np.linalg.norm(b))
        if verbose:
            tr = TRUTH[FN[f]]
            on = [(n, c) for n, c in zip(R["terms"], w) if c != 0]
            errs = [abs(c - tr[n]) for n, c in on if n in tr]
            print(f"\nd{FN[f]}/dt =" + "".join(f" {'+' if c >= 0 else '-'}{abs(c):.4f}*{n}" for n, c in on))
            print(f"   lambda={lam:.2e}  rel. residual={R['res'][-1]:.2e}   true terms found {len(errs)}/{len(tr)}, "
                  f"spurious {len(on) - len(errs)}, max coeff error on true terms {max(errs, default=0):.4f}")
    return R


def write_csvs(R, prefix):
    with open(f"{prefix}_coefs.csv", "w") as oc:
        oc.write("target,term,coef,truth,lambda,relres\n")
        for f in range(6):
            for n, c in zip(R["terms"], R["W"][f]):
                oc.write(f"{FN[f]},{n},{c:.8e},{float(TRUTH[FN[f]].get(n, 0.0)):.8e},"
                         f"{R['lam'][f]:.8e},{R['res'][f]:.8e}\n")
    with open(f"{prefix}_fit.csv", "w") as of:               # weak-form fit: b = (phi,u_t) and G*w per window
        of.write("row" + "".join(f",b_{n},Gw_{n}" for n in FN) + "\n")
        for r in range(R["B"][0].size):
            of.write(str(r) + "".join(f",{R['B'][f][r]:.6e},{R['GW'][f][r]:.6e}" for f in range(6)) + "\n")


def to_rows(R):
    """WSINDy result -> same structure as read_coefs."""
    rows = {FN[f]: [(n, float(c), float(TRUTH[FN[f]].get(n, 0.0))) for n, c in zip(R["terms"], R["W"][f])]
            for f in range(6)}
    return rows, dict(zip(FN, R["res"]))


def read_coefs(path):
    """-> dict target -> list of (term, coef, truth), plus dict target -> relres."""
    rows, res = {}, {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["target"], []).append((r["term"], float(r["coef"]), float(r["truth"])))
            res[r["target"]] = float(r["relres"])
    return rows, res


def score(rows):
    """max error on true terms (missing term = full error), #spurious, #missed, over all 6 equations."""
    err, spur, miss = [], 0, 0
    for tg in FN:
        for _, c, tv in rows[tg]:
            if tv != 0:
                err.append(abs(c - tv)); miss += (c == 0)
            elif c != 0:
                spur += 1
    return (max(err) if err else np.nan), spur, miss


def cmd_fit(a):
    F, h = load_data(a.npz)
    print(f"data: nt,nx,nz = {','.join(map(str, F.shape[1:]))}  dt,dx,dz = {h[0]:.5g},{h[1]:.5g},{h[2]:.5g} um")
    R = wsindy(F, h, a.noise, a.seed, a.max_deriv, a.quadratic)
    write_csvs(R, a.prefix)
    print("\nexpected:\n  dEx/dt = -By_z   dEy/dt = +Bx_z -Bz_x   dEz/dt = +By_x\n"
          "  dBx/dt = +Ey_z   dBy/dt = +Ez_x -Ex_z   dBz/dt = -Ey_x")
    print(f"\nsaved {a.prefix}_coefs.csv and {a.prefix}_fit.csv")


# =============================================================== 4. plots
class Figs:
    """Saves each figure as a PNG and collects all of them into one PDF."""
    def __init__(self, a):
        import matplotlib
        if not a.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        self.plt, self.show = plt, a.show
        self.out = Path(a.outdir); self.out.mkdir(exist_ok=True)
        self.pdf_path = self.out / "all_plots.pdf"
        self.pdf, self.saved = PdfPages(self.pdf_path), []

    def save(self, fig, name):
        fig.savefig(self.out / name, dpi=130)
        self.pdf.savefig(fig)
        self.saved.append(name)

    def close(self):
        self.pdf.close()
        print(f"saved {', '.join(self.saved)} in {self.out.resolve()}")
        print(f"all {len(self.saved)} figures together: {self.pdf_path}")
        if self.show:
            self.plt.show()


def plot_recovery(figs, rows, res, fit_path):
    plt = figs.plt
    from matplotlib.patches import Rectangle

    # bar charts: recovered vs true coefficients
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, tg in zip(axes.flat, FN):
        show = [(t, c, tv) for t, c, tv in rows[tg] if c != 0 or tv != 0]
        x, wd = np.arange(len(show)), 0.38
        ax.bar(x - wd / 2, [tv for *_, tv in show], wd, color="0.65", label="true")
        ax.bar(x + wd / 2, [c for _, c, _ in show], wd, color="C0", label="recovered")
        ax.set_xticks(x); ax.set_xticklabels([t for t, *_ in show], rotation=30, ha="right")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlim(-1, max(len(show), 3)); ax.set_ylim(-1.35, 1.35)
        ax.set_title(f"d{tg}/dt   (rel. residual {res[tg]:.1e})", fontsize=10)
        for xi, (t, c, tv) in zip(x, show):
            if tv == 0:
                ax.annotate("spurious", (xi + wd / 2, c), ha="center", va="bottom", fontsize=7, color="C3")
    axes[0, 0].legend()
    fig.suptitle("WSINDy recovery of vacuum Maxwell: coefficients (units: E/c, x and ct in um)")
    figs.save(fig, "recovery_bars.png")

    # heatmap of the whole coefficient matrix
    terms = [t for t, *_ in rows["Ex"]]
    M = np.array([[c for _, c, _ in rows[tg]] for tg in FN])
    T = np.array([[tv for *_, tv in rows[tg]] for tg in FN])
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(terms)), 4.5), constrained_layout=True)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    ax.set_xticks(range(len(terms))); ax.set_xticklabels(terms, rotation=60, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels([f"d{t}/dt" for t in FN])
    for i, j in zip(*np.nonzero(T)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=2))
    for i, j in zip(*np.nonzero(M)):
        ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="coefficient")
    ax.set_title("Recovered coefficients (black boxes = true terms)")
    figs.save(fig, "recovery_heatmap.png")

    # weak-form fit, one point per window
    if not Path(fit_path).exists():
        print(f"({fit_path} not found, skipping fit plot)")
        return
    d = np.genfromtxt(fit_path, delimiter=",", names=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, tg in zip(axes.flat, FN):
        b, gw = d[f"b_{tg}"], d[f"Gw_{tg}"]
        k = slice(None, None, max(1, len(b) // 8000))
        ax.plot(b[k], gw[k], ".", ms=2, alpha=0.4)
        lim = np.abs(b).max() * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
        ax.set_xlabel(f"(phi, d{tg}/dt)  from data"); ax.set_ylabel("G w  from recovered model")
        ax.set_title(f"{tg}: rel. residual {res[tg]:.1e}", fontsize=10)
    fig.suptitle("Weak-form fit, one point per test-function window (on the line = perfect)")
    figs.save(fig, "recovery_fit.png")


def plot_noise(figs, a):
    """One field, one snapshot, at several noise ratios, plus a 1D cut through the middle."""
    plt = figs.plt
    F, (_, dx, _) = load_data(a.npz)
    f = FN.index(a.field)
    u = F[f]                                                 # (t, x, z)
    nt, nx, nz = u.shape
    snap = min(a.snap, nt - 1)
    rms = np.sqrt(np.mean(u ** 2))                           # RMS over all space-time, as in the fit
    unit = noise_field(u.shape, 1.0, 1.0, a.seed, f)[snap]   # the fit's own noise draw for this field
    clean = u[snap]
    vmax = np.abs(clean).max()                               # one colour scale for every panel
    ext = [a.x0, a.x0 + nx * dx] * 2

    fig, ax = plt.subplots(1, len(a.ratios), figsize=(3.3 * len(a.ratios), 3.8), constrained_layout=True)
    ax = np.atleast_1d(ax)
    for k, r in enumerate(a.ratios):
        im = ax[k].imshow((clean + r * rms * unit).T, origin="lower", extent=ext, cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
        ax[k].set_title(f"{100 * r:g}% noise", fontsize=11)
        ax[k].set_xlabel("x (um)")
    ax[0].set_ylabel("z (um)")
    fig.colorbar(im, ax=ax, shrink=0.85, label=f"{a.field} (T)")
    fig.suptitle(f"{a.field}, snapshot {snap}: noise std = ratio x RMS of {a.field} over the whole run "
                 f"(RMS = {rms:.3g})", fontsize=11)
    figs.save(fig, "noise_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    xs = a.x0 + (np.arange(nx) + 0.5) * dx
    for r in a.ratios:
        ax.plot(xs, (clean + r * rms * unit)[:, nz // 2], lw=1, label=f"{100 * r:g}%")
    ax.plot(xs, clean[:, nz // 2], "k", lw=2, label="clean")
    ax.set_xlabel("x (um)"); ax.set_ylabel(a.field); ax.legend(ncol=3, fontsize=8)
    ax.set_title(f"{a.field}, snapshot {snap}, cut at the middle z")
    figs.save(fig, "noise_slice.png")
    print(f"RMS of {a.field} over the run = {rms:.4g}; snapshot {snap} has max |{a.field}| = {vmax:.4g}")


def plot_sweep(figs, a):
    """Rerun WSINDy at each noise level (several seeds) and plot how recovery degrades."""
    plt = figs.plt
    F, h = load_data(a.npz)
    levels = list(a.sweep)
    E, S, M, R = ([[] for _ in levels] for _ in range(4))
    for i, lvl in enumerate(levels):
        for sd in range(a.seeds if lvl > 0 else 1):
            print(f"sweep: noise {lvl:g}, seed {1000 + sd}")
            rows, res = to_rows(wsindy(F, h, lvl, 1000 + sd, a.max_deriv, a.quadratic, verbose=False))
            e, s, m = score(rows)
            E[i].append(e); S[i].append(s); M[i].append(m); R[i].append(np.mean(list(res.values())))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    xs = np.arange(len(levels))
    for axk, data, title in zip(ax, [E, S, M, R],
                                ["max |coef - true| on true terms", "spurious terms (all 6 equations)",
                                 "true terms missed (all 6 equations)", "mean relative residual"]):
        for i, d in enumerate(data):
            axk.plot([xs[i]] * len(d), d, "o", color="C0", alpha=0.4)
        axk.plot(xs, [np.mean(d) for d in data], "-o", color="C0")
        axk.set_xticks(xs); axk.set_xticklabels([f"{v:g}" for v in levels])
        axk.set_xlabel("noise ratio  (std / RMS of each field)"); axk.set_title(title, fontsize=10)
        axk.grid(alpha=0.3)
    ax[0].set_yscale("log"); ax[3].set_yscale("log")
    fig.suptitle(f"WSINDy noise sweep  (max_deriv={a.max_deriv}, quadratic={a.quadratic})")
    figs.save(fig, "recovery_noise_sweep.png")


def plot_all(figs, a):
    cpath = Path(f"{a.prefix}_coefs.csv")
    if not cpath.exists():
        sys.exit(f"{cpath} not found. Run:  python {Path(__file__).name} fit")
    rows, res = read_coefs(cpath)
    plot_recovery(figs, rows, res, f"{a.prefix}_fit.csv")
    plot_noise(figs, a)
    for tg in FN:
        print(f"d{tg}/dt =", " ".join(f"{c:+.4f}*{t}" for t, c, _ in rows[tg] if c != 0) or "0")


def cmd_plot(a):
    figs = Figs(a)
    plot_all(figs, a)
    figs.close()


def cmd_sweep(a):
    if not a.sweep:
        sys.exit("give the noise levels, e.g.  --sweep 0 0.01 0.05 0.1")
    figs = Figs(a)
    plot_sweep(figs, a)
    figs.close()


def cmd_all(a):
    if a.rerun or not Path(a.npz).exists():
        cmd_run(a)
        cmd_convert(a)
    else:
        print(f"using existing {a.npz} (add --rerun to run WarpX again)")
    cmd_fit(a)
    figs = Figs(a)
    plot_all(figs, a)
    if a.sweep:
        plot_sweep(figs, a)
    figs.close()


# =============================================================== command line
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--npz", default=str(HERE / "maxwell_vacuum.npz"), help="field data (.npz, or old .bin)")
    common.add_argument("--prefix", default="wsindy", help="csv names: <prefix>_coefs.csv, <prefix>_fit.csv")
    common.add_argument("--outdir", default="plots")
    common.add_argument("--show", action="store_true", help="open plot windows")
    common.add_argument("--seed", type=int, default=12345, help="noise seed (fit and noise plot)")
    common.add_argument("--max-deriv", type=int, default=1, help="highest spatial derivative in the library")
    common.add_argument("--quadratic", type=int, default=0, help="1 = add products Ex*Ey etc. to the library")
    common.add_argument("--noise", type=float, default=0.0, help="noise ratio for fit")
    common.add_argument("--sweep", nargs="+", type=float, help="noise ratios for the sweep")
    common.add_argument("--seeds", type=int, default=3, help="seeds per noise level in the sweep")
    common.add_argument("--field", default="Ey", choices=FN, help="field for the noise plot")
    common.add_argument("--snap", type=int, default=50, help="snapshot for the noise plot")
    common.add_argument("--ratios", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1.0],
                        help="noise ratios shown in the noise plot")
    common.add_argument("--x0", type=float, default=-128.0, help="left edge of the box (um), for plot axes")
    common.add_argument("--rerun", action="store_true", help="'all': run WarpX even if the .npz exists")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, hlp in [("run", cmd_run, "run WarpX"), ("convert", cmd_convert, "openPMD -> .npz"),
                          ("fit", cmd_fit, "WSINDy fit"), ("plot", cmd_plot, "recovery + noise plots"),
                          ("sweep", cmd_sweep, "noise sweep plot"), ("all", cmd_all, "everything")]:
        sub.add_parser(name, parents=[common], help=hlp).set_defaults(func=fn)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()