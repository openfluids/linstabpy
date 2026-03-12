# Errors I Had

## Summary

After running `bash install.sh` and then `source .venv/bin/activate` and `python example_6_hamada.py`, the run failed with:

```text
ImportError: libslepc.so.3.24: cannot open shared object file: No such file or directory
```

The install script reported **“All checks passed”**, so the failure was surprising.

---

## What I Observed

1. **Install completed successfully**
   - `bash install.sh` ran to completion.
   - PETSc, SLEPc, petsc4py, and slepc4py were installed (versions 3.24.2).
   - The verification block at the end printed “All checks passed” for Python, numpy, scipy, matplotlib, petsc4py, slepc4py, mpi4py, and linstab2d.

2. **Running the example failed**
   - I activated the venv: `source .venv/bin/activate`.
   - I ran: `python example_6_hamada.py`.
   - The failure happened when importing `linstab2d.solvers.eigenvalue`, which pulls in `linstab2d.solvers.resolvent`, which calls `slepc4py.init()`. At that point the dynamic linker could not find `libslepc.so.3.24`.

3. **Why verification passed but the script failed**
   - During `install.sh`, the script sets `LD_LIBRARY_PATH` in the **same shell** (line 275) to include both PETSc and SLEPc lib directories. The verification Python snippet runs in that shell, so `from slepc4py import SLEPc` succeeds and “All checks passed” is printed.
   - The script then writes only **PETSc’s** lib path into `.venv/bin/activate` (lines 316–319). It does **not** add SLEPc’s lib path there.
   - When I later run `source .venv/bin/activate` and `python example_6_hamada.py`, `LD_LIBRARY_PATH` only contains PETSc’s lib directory. The loader therefore cannot find `libslepc.so.3.24`, and the import fails.

So the root cause is: **SLEPc’s library directory is never persisted into the venv activate script on Linux**, so after a fresh shell + activate, the loader doesn’t know where to find `libslepc.so.3.24`.

---

## Fix (for maintainers)

In `install.sh`, when appending to `.venv/bin/activate`, include **both** PETSc and SLEPc lib directories in `LD_LIBRARY_PATH`, for example:

```bash
# Persist the setting in the virtual environment activation script
if ! grep -q "PETSC_DIR/lib" "${VENV_DIR}/bin/activate" 2>/dev/null; then
    echo "export LD_LIBRARY_PATH=\"${PETSC_DIR}/lib:${SLEPC_DIR}/lib:\${LD_LIBRARY_PATH}\"" >> "${VENV_DIR}/bin/activate"
fi
```

After re-running the install (or manually adding the SLEPc path to `activate`), `source .venv/bin/activate` and `python example_6_hamada.py` should work.

---

## Workaround (without re-running install)

If you don’t want to change the install script yet, after activating the venv set the library path yourself. For example, if SLEPc is in the venv site-packages:

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH="${SLEPC_DIR:-.venv/lib/python3.13/site-packages/slepc}/lib:${PETSC_DIR:-.venv/lib/python3.13/site-packages/petsc}/lib:$LD_LIBRARY_PATH"
python example_6_hamada.py
```

Or discover the path and set it once:

```bash
SLEPC_DIR=$(python -c "import slepc; print(slepc.get_slepc_dir())")
export LD_LIBRARY_PATH="${SLEPC_DIR}/lib:$LD_LIBRARY_PATH"
python example_6_hamada.py
```

---

## MATLAB `.mat` sponge file error (v7.3 vs `loadmat`)

After fixing the SLEPc loader issue, running:

```bash
source .venv/bin/activate
python example_6_hamada.py
```

failed during the Hamada baseflow load with:

```text
NotImplementedError: Please use HDF reader for matlab v7.3 files, e.g. h5py
```

### What this means

- The Hamada loader (`linstab2d/baseflow/hamada.py`) was modified earlier to remove the hard dependency on `h5py` and to use `scipy.io.loadmat` in `_load_sponge_aux()` instead.
- `loadmat` can only read **MATLAB v5/v7 (non‑HDF5)** `.mat` files.  
  For **v7.3 (HDF5) `.mat` files**, SciPy raises exactly this `NotImplementedError` and recommends using `h5py` instead.
- The file
  ```text
  example_6_hamada/MATLAB_exports_resolvent/case_inputs_used.mat
  ```
  in this setup is a **v7.3** `.mat`, so `loadmat` cannot read it.
- `install.sh` does **not** install `h5py`, so there is no HDF5 reader available in the venv.

### Why this is acceptable for my current use

In `example_6_hamada.py`, the `.mat` file is only used to load an alternative `sponge_aux` field for strict **parity with MATLAB exports**. For my current runs I:

- Do not need to reproduce the MATLAB sponge exactly.
- Am happy to use the sponge array already stored in the `.npy` baseflow files.

Because of that, instead of re‑adding `h5py` and HDF5 handling, I changed the example so that:

- **By default, it does not use the MATLAB `.mat` sponge at all.**  
- It calls `load_hamada_case(...)` with `prefer_matlab_sponge_aux=False`, so `_load_sponge_aux()` is never triggered and the code relies entirely on the `.npy` sponge field.

This avoids both the `NotImplementedError` from `loadmat` and the need to install `h5py`, while still allowing parity checks in the future if I explicitly re‑enable the MATLAB sponge path.
