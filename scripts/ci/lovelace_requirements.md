# Lovelace (Cenapad) — modules for `install_tester`

Oracle Linux 8 + OpenHPC-style stack (`/opt/ohpc`). Interactive nodes such as `lovelace.ada.cenapad.unicamp.br` typically **do not have `sudo`**. Load the modules below so `h5cc` and MPI wrappers are available; then skip distro package installs.

## Required `module load` (verified workflow)

```bash
module purge   # optional: reduces mixed MPI/compiler surprises

module load gnu9
module load cmake/3.21.3
module load mvapich2
module load autotools
module load hdf5/1.12.1-mvapich2
```

- **`gnu9`**: GCC 9.4.0 toolchain (`gcc`, `g++`, `gfortran`).
- **`cmake/3.21.3`**: CMake from OHPC (meets `install.sh` expectations).
- **`mvapich2`**: MPI in the gnu9 stack (Hydra `mpiexec` / `mpicc` under `/opt/ohpc/pub/mpi/mvapich2-gnu9/...`).
- **`autotools`**: helps satisfy build-helper expectations where needed.
- **`hdf5/1.12.1-mvapich2`**: provides **`h5cc`** and HDF5 headers/libs **built against the same MVAPICH2** as above. Without it, `install.sh` may try `sudo dnf install ...` for `hdf5-devel`, which fails for normal users.

If a module name differs on your node, run:

```bash
module avail hdf5 2>&1 | grep -i mvapich
module spider hdf5
```

and pick the **HDF5 variant that matches `mvapich2`**, not a generic or OpenMPI build.

## Install commands (no sudo)

After `module load`:

```bash
bash diagnose.sh
export LINSTABPY_SKIP_SYS_DEPS=1
export LINSTABPY_PETSC_MPIEXEC="$(command -v mpiexec.hydra 2>/dev/null || command -v mpiexec)"
bash run_install_tester.sh lovelace
# or: bash install.sh
```

- **`LINSTABPY_SKIP_SYS_DEPS=1`**: do not run `sudo dnf`; rely on modules + `uv` for Python.
- **`LINSTABPY_PETSC_MPIEXEC`**: optional but useful so PETSc configure uses the same Hydra `mpiexec` as diagnostics (`diagnose.sh` prints a hint when probes pass).

For HDF5-enabled Python extras (only if you need h5py):

```bash
LINSTABPY_WITH_H5PY=1 bash install.sh
```

(requires a consistent HDF5 module such as the one above).

## Quick checks

```bash
which mpicc mpiexec h5cc cmake
mpicc -show
diagnose.sh    # should report ALL CHECKS PASSED; h5cc OK if HDF5 module is loaded
```

## `libmpi.so: not found in ldconfig cache`

Common on module-only MPI installs. If runtime errors mention missing `libmpi`, prepend the MVAPICH2 `lib` directory to `LD_LIBRARY_PATH` (path is under `/opt/ohpc/pub/mpi/mvapich2-gnu9/.../lib` after `module load mvapich2`; use `echo "$LD_LIBRARY_PATH"` after loading modules).

## Note on `install.sh` and `h5cc`

The installer currently treats `h5cc` as a prerequisite when missing and may attempt `dnf` before you can opt out. On shared clusters, always set **`LINSTABPY_SKIP_SYS_DEPS=1`** *or* load an HDF5 module first so `h5cc` is present.
