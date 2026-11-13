# Use linstabpy with Claude Code

[Claude Code](https://code.claude.com/docs/en/overview) is an AI coding assistant that runs in your terminal: it reads files, runs commands and edits code in the folder where you start it. The repository ships an *agent skill* for it, the folder [`skills/linstabpy/`](https://github.com/openfluids/linstabpy/tree/main/skills/linstabpy). A skill is a set of instructions that Claude Code loads when your request matches it. This one teaches Claude how linstabpy is installed, how a case is defined and run, and how results are stored, so it does not guess.

This page shows one session from start to finish: install the skill, build the solver stack, run the examples, run your own case, read the results. Each step gives a request you can type, the commands the skill tells Claude to run, and the output those commands produced on a workstation (AMD Ryzen 9 9900X, Ubuntu 24.04 under WSL2, Intel MPI 2021.19, PETSc 3.24.2, Python 3.14.9, linstabpy at commit `6278e74`, 2026-11-13). Time and memory differ between machines; gains and eigenvalues agree to many digits.

## 1. Install the skill

The skill is in the git repository. It is not in the package that `pip install linstabpy` installs, so clone the repository first. Then link the skill folder into your Claude Code skills folder, from the repository root:

```bash
git clone https://github.com/openfluids/linstabpy.git
cd linstabpy
mkdir -p ~/.claude/skills
ln -s "$PWD/skills/linstabpy" ~/.claude/skills/linstabpy
```

This makes the skill available in every folder where you start Claude Code. To make it available only inside one project, link it into that project's `.claude/skills/` folder instead.

Start Claude Code in the repository (`claude`). A running session picks up a new skill without a restart. To check that it loaded, type `/skills`: the list shows `linstabpy`. Claude loads the skill on its own when a request mentions linstabpy, eigenvalues, resolvent gain, the Chu norm or a YAML case. You can also call it directly by typing `/linstabpy` before your request.

## 2. Build the solver stack

Request:

> Install linstabpy on this machine and check that the solver stack works.

The skill makes the install check the first step, because `import linstabpy` succeeds even when PETSc is broken. The commands are:

```bash
bash install.sh --diagnose   # reads the machine, installs nothing
bash install.sh              # builds .venv: complex PETSc/SLEPc with MUMPS, mpi4py, linstabpy
bash install.sh --check      # import, complex scalars, a small eigenvalue solve, a 2-rank MPI test
```

The last lines of `--check` on the test machine:

```text
python=3.14.9
petsc4py=3.24.2, scalar=complex
slepc4py=3.24.2
All checks passed.
[install] Multi-rank verification will use: /opt/intel/oneapi/mpi/2021.19/bin/mpiexec -n 2
mpi4py_multirank=size=2
petsc4py_multirank=size=2
petsc4py_multirank_vec_sum=(6+0j)
petsc4py_multirank_ksp=OK
```

Nothing that solves will work until `--check` passes. If it fails, ask Claude to read the log with you; the skill points it to the troubleshooting notes (mixed MPI installations, the macOS BLAS problem, thread settings).

## 3. Smoke test and first real run

Request:

> Run the smallest example, then the Blasius resolvent example, and tell me the gains.

The skill runs example 0 first. It needs no data and no MPI, and it proves the Python side works:

```bash
.venv/bin/python -m examples.linstab2d.example_0_curvilinear_validation
```

```text
Saved fields: .../examples/linstab2d/example_0_curvilinear_validation.npz
example_0_curvilinear_validation  0.122 s
```

Then the Blasius boundary layer (Re = 6.0e5, Ma = 0.3, 300 x 100 grid), which reads the base flow shipped in `examples/linstab2d/baseflows/`:

```bash
.venv/bin/python -m examples.linstab2d.example_1_blasius
```

```text
Mode           sigma       rho_fwd       rho_adj
--------------------------------------------------
   1         14.6571    1.0142e-16    8.5993e-14
   2         1.89912    8.3096e-17    1.5029e-13
   3         1.34278    1.0059e-16    2.3366e-13
                 max    1.0142e-16    2.3366e-13
Blasius timing: build_lnse_operator=1.83s, resolvent_analysis=27.59s
Resolvent gains: [14.65714004  1.89911838  1.34277958]
example_1_blasius  32.680 s
```

One process, 33 s, 2.3 GB peak memory. `sigma` is the resolvent gain; `rho_fwd` and `rho_adj` are the residuals of the singular-value problem, and both must be small. The figures come from `.venv/bin/python -m examples.linstab2d.example_1_plots`.

## 4. Your own case through the `linstabpy` command

Request:

> Copy examples/sawtooth/sawtooth_evp.yaml to sawtooth_coarse.yaml and make it cheap: every fourth grid point in both directions, 6 modes, a Krylov subspace of 40. I only want to see the pipeline work. Check the file, then run it with the linstabpy command.

The `linstabpy` command takes no case arguments. The case file and the output file travel in two environment variables. The steps below are the commands for this request; the output is from running them on the test machine.

1. The four changes in the copied file: `numerics.stride_ksi: 4` and `numerics.stride_eta: 4`, `analysis.n_modes_stability: 6`, `solver.eigen.ncv: 40`. Without a file to start from, the skill starts from `skills/linstabpy/assets/minimal-case.yaml`.
2. Check the file before running it. `validate_config` checks the keys and values; `check_cli_ignored_keys` refuses flat keys such as `Re` that only the example scripts read:

    ```bash
    .venv/bin/python -c "
    from linstabpy.config.schema import validate_config, check_cli_ignored_keys
    from linstabpy.utils.yaml_io import load_config
    cfg = load_config('sawtooth_coarse.yaml')
    validate_config(cfg, source='sawtooth_coarse.yaml')
    check_cli_ignored_keys(cfg, source='sawtooth_coarse.yaml')"
    ```

3. Run it, one thread per process:

    ```bash
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export LINSTABPY_CASE_CONFIG=sawtooth_coarse.yaml
    export LINSTABPY_CASE_OUTPUT_NPZ=sawtooth_coarse.npz
    .venv/bin/linstabpy            # one process; for MPI: mpiexec -n 4 .venv/bin/linstabpy (not run here)
    ```

Part of the log (excerpt):

```text
[linstabpy] preflight advisories: 1
[linstabpy]   [note] STRIDE-NOT-FINAL: strided mesh (stride_ksi=4, stride_eta=4): this is a reduced-resolution run, valid as ladder/parity/cost evidence but not as a final result
[linstabpy] loaded case arrays: shape=(85, 75), points=6375, sponge_source=off, sponge_max=0, sponge_nnz=0
[linstabpy]   [WARNING] BASEFLOW-NORMALISATION: no part of the base flow has rho = u = T = 1, so it does not follow the canonical freestream convention; ...
[linstabpy] operator assembled with BCs: shape=(31875, 31875), nnz=3021694
[linstabpy] running temporal eigensolve on i*L0 (MATLAB convention): target=0.561107-1.112186j, modes=6
[linstabpy] stability eig convergence: nev=6 ncv=40 nconv=6 eps_iterations=1 max_residual=1.071e-05 median_residual=7.982e-06
[linstabpy] stability postcheck: max_rel=3.339e-05 median_rel=1.853e-05
[linstabpy] results saved to .../sawtooth_coarse.npz
[linstabpy] compute finished in 18.05s
```

The run printed two advisories. The first says the coarse grid makes this a test run, not a result. The second says the base flow is not scaled to the free-stream values linstabpy expects. Ask Claude to report every advisory to you and not to skip past them; a mis-scaled base flow invalidates every eigenvalue from the run. This run shows the pipeline working. Its eigenvalues are not a physical result.

The case files next to the Martini examples (`examples/linstab2d/example_1_blasius.yaml` and the others) are read by the example scripts, not by the `linstabpy` command. They pass `validate_config` but not `check_cli_ignored_keys`, and the command refuses them with the same message, naming each key to move:

```text
linstabpy.config.schema.ConfigValidationError: examples/linstab2d/example_1_blasius.yaml: top-level 'Re' (use physics.Re), 'Ma' (use physics.Ma), ... is not read by the linstabpy command; set the value under the nested key
```

For a new case, start from `skills/linstabpy/assets/minimal-case.yaml` or from a case under `examples/sawtooth/` or `examples/savarino/`.

## 5. Read the results

Request:

> Which eigenvalues did the run find, and is any of them unstable?

Results are stored in one `.npz` file whose key names stay fixed between versions. It reads with plain NumPy:

```python
import numpy as np
data = np.load("sawtooth_coarse.npz")
omega = data["omega"]              # temporal eigenvalues
modes = data["direct_modes"]       # eigenvectors, one column per mode
print(omega)
print("unstable:", int((omega.imag > 0).sum()), "of", omega.size)
```

```text
[ 0.5463-1.4821j -0.5463-1.4821j  1.4272-1.9725j  1.0752-2.9235j
 -1.4272-1.9725j -1.0752-2.9235j]
unstable: 0 of 6
```

The time dependence is `exp(-i omega t)`, so a mode grows exactly when `Im(omega) > 0`. The eigenvalues come in pairs `omega` and `-conj(omega)` because the operator is real. The full list of keys is on the [runtime NPZ contract](development/runtime-npz-contract.md) page.

## 6. Mistakes the skill prevents

| Mistake | What happens without the skill | What the skill makes Claude do |
|---|---|---|
| Inventing command-line arguments, such as `linstabpy run case.yaml` | The command prints `warning: ignoring unknown arguments ['run', 'case.yaml']` and runs whatever case the environment variables name, with exit code 0. | Set `LINSTABPY_CASE_CONFIG` and `LINSTABPY_CASE_OUTPUT_NPZ`; never add flags. |
| Rerunning after a change and getting old results | If the output file exists with the same config and code version, the command prints `Cache valid (...), using cached results.` and skips the computation. | Use one output file per run, or set `LINSTABPY_FORCE_RECOMPUTE=1` on purpose. |
| `uv run` on macOS | `uv run` can skip the BLAS layout that `install.sh` set up, and the eigenvalue solver aborts. | Always call `.venv/bin/python` or `.venv/bin/linstabpy`. |
| Mixing two MPI installations | Crashes or wrong-launcher errors at run time. | One MPI for the launcher, mpi4py, PETSc and SLEPc; rebuild rather than patch `LD_LIBRARY_PATH`. |
| Many threads per MPI rank | Each rank starts a full set of BLAS threads and the machine slows down. | Export one thread per rank before `mpiexec`. |

## When the repository moves

The `.venv` that `install.sh` builds stores absolute paths. If you move or rename the repository folder, `bash install.sh --check` fails with

```text
ImportError: libpetsc.so.3.24: cannot open shared object file: No such file or directory
```

Run `bash install.sh` again in the new place, then `bash install.sh --check`.

## Without the skill

An assistant that cannot load skills, such as a chat in a web browser, can read the documentation directly. Give it [llms.txt](https://linstabpy.vercel.app/llms.txt), an index of this site written for language models, or [llms-full.txt](https://linstabpy.vercel.app/llms-full.txt), which holds all pages in one file, and then ask your question.
