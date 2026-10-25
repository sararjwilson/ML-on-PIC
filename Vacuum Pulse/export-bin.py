"""npz -> flat binary for the C++ WSINDy code. Lengths and ct are converted to microns so the
Maxwell coefficients come out as +/-1.
Usage: python export-bin.py [maxwell_vacuum.npz] [maxwell_vacuum.bin]
Layout: int32 nt,nx,nz | float64 dt,dx,dz (um) | then Ex,Ey,Ez,Bx,By,Bz, each float64 C-order (t,x,z)."""
import sys
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "maxwell_vacuum.npz"
dst = sys.argv[2] if len(sys.argv) > 2 else "maxwell_vacuum.bin"
d = np.load(src)
t = d["t"] * 1e6
dt = float(t[1] - t[0])
assert np.allclose(np.diff(t), dt, rtol=1e-6), "snapshots are not uniformly spaced in time"
dx, dz = float(d["dx"]) * 1e6, float(d["dz"]) * 1e6
nt, nx, nz = d["Ex"].shape
with open(dst, "wb") as f:
    np.array([nt, nx, nz], dtype="<i4").tofile(f)
    np.array([dt, dx, dz], dtype="<f8").tofile(f)
    for k in ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]:
        np.ascontiguousarray(d[k], dtype="<f8").tofile(f)
print(f"wrote {dst}: nt,nx,nz = {nt},{nx},{nz}  dt,dx,dz = {dt:.4g},{dx:.4g},{dz:.4g} um")
