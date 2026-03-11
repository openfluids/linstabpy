# What is saved in timing_and_memory.mat

The file `timing_and_memory.mat` is written by the instrumented `main.m` script. It contains per-step timing and memory usage from the last run.

## Variables

| Variable       | Type        | Description |
|----------------|-------------|-------------|
| `step_names`   | cell array  | Step identifiers in order of execution. Possible names: `setup_parameters`, `create_mesh`, `base_flow`, `sponge`, `linear_operator_BCs`, `resolvent` (if run), `stability_analysis` (if run). |
| `step_times`   | double array| Elapsed time in **seconds** for each step (same order as `step_names`). |
| `step_mem_MB`  | double array| Process heap memory in **megabytes** after each step (same order as `step_names`). Measured via Java Runtime; reflects MATLAB process memory. |

## Step meanings

- **setup_parameters**  Setting physical, EVP, and eigs options.
- **create_mesh**  Loading mesh, CreateMesh, DeformMesh.
- **base_flow**  readbaseflow and sutherland_air.
- **sponge**  Loading and applying sponge.
- **linear_operator_BCs**  GetLinProblem, BC_Dirichlet/BC_Neumann, GetCompEnergyNorm, saving idx.
- **resolvent**  Resolvent analysis (only if `resolvent_flag` is true).
- **stability_analysis**  eigs and saving spectrum/eigenvectors (only if `stability_analysis_flag` is true).
