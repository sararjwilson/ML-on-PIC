from openpmd_viewer import OpenPMDTimeSeries
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# run warpx.nd inputfile first

# plot phase space using opemPMD-viewer
script_dir = os.path.dirname(os.path.abspath(__file__))
ts = OpenPMDTimeSeries(os.path.join(script_dir, "diags", "diag1"))

# animation loop set-up
frames = []
for it in ts.iterations:
    per_species = []
    for sp in ts.avail_species:
        z, uz = ts.get_particle(['z', 'uz'], species=sp, iteration=it)
        per_species.append((z * 1e6, uz)) # z in micrometers
    frames.append(per_species)

# fixed limits
all_uz = np.concatenate([uz for per_species in frames for _, uz in per_species])
uz_lim = 1.05 * np.max(np.abs(all_uz))
z_all = np.concatenate([z for z, _ in frames[0]])

# plots
fig, ax = plt.subplots(figsize=(7,5))
scatters = [ax.scatter([], [], s=0.5, alpha=0.3, label = sp) for sp in ts.avail_species]
ax.set_xlim(np.min(z_all), np.max(z_all))
ax.set_ylim(-uz_lim, uz_lim)
ax.set_xlabel(r'z [$\mu$m]')
ax.set_ylabel(r'$u_z$ [c]')
ax.legend(markerscale=20, loc='upper right')
title = ax.set_title('')

def update(k):
    for sc, (z, uz) in zip(scatters, frames[k]):
        sc.set_offsets(np.column_stack([z, uz]))
    title.set_text(f"Iteration: {ts.iterations[k]}")
    return scatters + [title]
anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
anim.save(os.path.join(script_dir, "phase_space_animation.mp4"), dpi=200, writer='ffmpeg')

# midpoint animation snapshot to get usable figure
mid_k = len(frames) // 2
update(mid_k)
fig.savefig(
    os.path.join(script_dir, "phase_space_snapshot.png"),
    dpi=300,
    bbox_inches='tight'
)

