# Run WarpX 2D vacuum pulse; output lands in ./run/diags/diag1
import shutil, subprocess, sys
from pathlib import Path

here = Path(__file__).resolve().parent
inputs = here / "imputs-vacuumpulse.txt"      # your input file (note the filename spelling)
rundir = here / "run"

if not inputs.exists():
    sys.exit(f"Input file not found: {inputs}")

exe = shutil.which("warpx.2d") or shutil.which("warpx")
if exe is None:
    sys.exit("No WarpX executable found. Activate the env: conda activate warpx")

# start clean so stale snapshots don't get mixed in
if rundir.exists():
    shutil.rmtree(rundir)
rundir.mkdir()

print("Using:", exe)
subprocess.run([exe, str(inputs)], cwd=rundir, check=True)

diag = rundir / "diags" / "diag1"
print("Done. Snapshots in:", diag)
print(sorted(p.name for p in diag.iterdir())[:5], "...")

# conda activate warpx
# python run-vacuum.py