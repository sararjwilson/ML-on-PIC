import re
import numpy as np
import pandas as pd

def load_lumi(filename):
    with open(filename) as f:
        header = f.readline()
    E = np.array(list(map(float, re.findall("=(.*?)\\(", header))))  # in eV
    data = np.atleast_2d(np.loadtxt(filename))
    step, t = data[:, 0], data[:, 1]
    dLdE = data[:, 2:2 + len(E)]      # shape (n_times, n_bins)
    total = data[:, -1]
    return step, t, E, dLdE, total

d = "diags/reducedfiles/"
step, t, E, ep1, _ = load_lumi(d + "DiffLumi_ele1_pos2.txt")
_,    _, _, ep2, _ = load_lumi(d + "DiffLumi_pos1_ele2.txt")
ep = ep1 + ep2

# CSV: one row per energy bin, one column per time
df = pd.DataFrame(ep.T, index=E, columns=[f"t={ti:.4e}" for ti in t])
df.index.name = "E_eV"
df.to_csv("dLdE_ep.csv")

np.savez("lumi_ep.npz", E=E, t=t, u=ep.T)   # u[E_index, t_index]