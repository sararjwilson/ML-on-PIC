import numpy as np
import matplotlib
matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt

d = np.load("maxwell_vacuum.npz")
t, dx, dz = d["t"], float(d["dx"]), float(d["dz"])
F = {k: d[k] for k in ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]}   # shape (t, x, z)
nt, nx, nz = F["Bx"].shape
print("shape:", (nt, nx, nz), " dx, dz:", dx, dz, " dt (c*t units):", t[1] - t[0])

# 1. Blob locations at t=0 (x is axis 1, z is axis 2; grid spans -128..128 um)
x = -128e-6 + (np.arange(nx) + 0.5) * dx
z = -128e-6 + (np.arange(nz) + 0.5) * dz
for name in ["Bx", "By"]:
    i, j = np.unravel_index(np.abs(F[name][0]).argmax(), (nx, nz))
    print(f"{name} peak at t=0: x = {x[i]*1e6:.1f} um, z = {z[j]*1e6:.1f} um")
print("expected: Bx near x=-40, z=-20 +/- 14 um;  By near x=30 +/- 14, z=40 um")

# 2. Divergence of B (spectral), relative to the size of the individual derivatives
kx = 2*np.pi*np.fft.fftfreq(nx, dx)[None, :, None]
kz = 2*np.pi*np.fft.fftfreq(nz, dz)[None, None, :]
dBx = np.fft.ifft2(1j*kx*np.fft.fft2(F["Bx"], axes=(1, 2)), axes=(1, 2)).real
dBz = np.fft.ifft2(1j*kz*np.fft.fft2(F["Bz"], axes=(1, 2)), axes=(1, 2)).real
rel = np.sqrt(np.mean((dBx + dBz)**2, axis=(1, 2))) / np.sqrt(np.mean(dBx**2 + dBz**2, axis=(1, 2)))
print(f"relative div B: t=0 {rel[0]:.2e}, max over time {rel.max():.2e}  (should be small and not growing)")

# 3. Energy conservation
U = sum((F[k]**2).sum(axis=(1, 2)) for k in F)
print(f"energy drift: (max-min)/U0 = {(U.max() - U.min()) / U[0]:.2e}")

# 4. Plots
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for a, (name, k) in zip(ax.flat, [("Bx", 0), ("By", 0), ("Ey", 50), ("Ey", 200), ("Ex", 100), ("Ez", 100)]):
    im = a.imshow(F[name][k].T, origin="lower", extent=[x[0]*1e6, x[-1]*1e6, z[0]*1e6, z[-1]*1e6])
    a.set_title(f"{name}, snapshot {k}"); a.set_xlabel("x (um)"); a.set_ylabel("z (um)")
    plt.colorbar(im, ax=a)
plt.tight_layout(); plt.savefig("sanity_frames.png", dpi=120)

plt.figure(); plt.plot(U / U[0]); plt.xlabel("snapshot"); plt.ylabel("U / U0")
plt.title("Total field energy"); plt.savefig("sanity_energy.png", dpi=120)
print("saved sanity_frames.png, sanity_energy.png")
