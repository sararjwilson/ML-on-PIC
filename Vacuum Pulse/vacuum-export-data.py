# convert WarpX openPMD field output to maxwell_vacuum.npz (E/c, t -> c*t)
from pathlib import Path
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c

here = Path(__file__).resolve().parent
ts = OpenPMDTimeSeries(str(here / "run" / "diags" / "diag1"))
its = ts.iterations
print("snapshots:", len(its))

def load(field, coord):
    out = []
    for it in its:
        F, info = ts.get_field(field=field, coord=coord, iteration=it)
        axes = [info.axes[i] for i in range(F.ndim)]
        if axes == ["z", "x"]:
            F = F.T                      # force (x, z) ordering
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

np.savez(here / "maxwell_vacuum.npz",
         t=c * ts.t,
         dx=info.x[1] - info.x[0],
         dz=info.z[1] - info.z[0],
         **data)