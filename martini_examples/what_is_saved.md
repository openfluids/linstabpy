## Saved `.mat` files and their contents

This file summarizes all result and log files produced by the modified example scripts, and what each file contains.

### `setup.md` – benchmark machine configuration

The file **`setup.md`** is not a saved output of the examples; it documents the **MATLAB and hardware configuration** of the computer used to run the benchmarks. It records MATLAB version and toolboxes, operating system, CPU (model, cores, frequency), RAM, storage, and optionally GPU and power settings. Keeping this file up to date lets you compare timing and memory from the `*_logs.mat` files across different machines or after changing software/hardware.

---

### Example 0 – `example_0_TestCurvilinearDerivative.m`

- **Results**: `example_0_TestCurvilinearDerivative_results.mat`  
  - `results_TestCurv.mesh` – mesh struct used in the derivative test.  
  - `results_TestCurv.x`, `results_TestCurv.y` – grid coordinates.  
  - `results_TestCurv.f` – test function \(f = \cos(x)\).  
  - `results_TestCurv.df` – numerical derivative.  
  - `results_TestCurv.df_an` – analytical derivative \(-\sin(x)\).
- **Logs**: `example_0_TestCurvilinearDerivative_logs.mat`  
  - `timing` – struct array of steps with fields:
    - `timing(k).step` – `'mesh_CreateMesh_DeformMesh'`, `'derivative_computation'`, `'plot_derivatives'`.  
    - `timing(k).time` – time in seconds per step.  
  - `timing_total` – total runtime of the script.  
  - `mem_log` – struct array mirroring `timing`, with:
    - `mem_log(k).step` – step name.  
    - `mem_log(k).MB` – workspace memory in MB after that step.

### Example 1 – `example_1_Blasius.m`

- **Results**: `example_1_Blasius_results.mat`  
  - `results_Blasius.mesh` – computational/physical mesh.  
  - `results_Blasius.baseFlow` – Blasius base-flow fields and properties.  
  - `results_Blasius.gains` – resolvent gains.  
  - `results_Blasius.responses` – resolvent response modes.  
  - `results_Blasius.forces` – resolvent forcing modes.
- **Logs**: `example_1_Blasius_logs.mat`  
  - `timing` – steps:
    - `'mesh_CreateMesh_DeformMesh'` – mesh build + mapping.  
    - `'baseFlow_example1read_sutherland'` – base-flow construction + `sutherland_air`.  
    - `'sponge_build'` – sponge field creation.  
    - `'operator_GetLin_BC_CompEnergy'` – `GetLinProblem`, `BC_Dirichlet`, `GetCompEnergyNorm`.  
    - `'resolvent'` – resolvent assembly and solve.  
    - `'plotting'` – resolvent gain/mode plots (only if `verbose=true`).  
    - each with `time` in seconds.  
  - `timing_total` – total runtime.  
  - `mem_log` – same step names as `timing`, with memory in MB.

### Example 2 – `example_2_Jet.m`

- **Results**: `example_2_Jet_results.mat`  
  - `results_Jet.mesh` – axisymmetric jet mesh (computational and physical).  
  - `results_Jet.baseFlow` – jet base-flow fields built from LES data.  
  - `results_Jet.S` – resolvent gains.  
  - `results_Jet.U` – resolvent response modes.  
  - `results_Jet.V` – resolvent forcing modes.
- **Logs**: `example_2_Jet_logs.mat`  
  - `timing` – steps:
    - `'mesh_CreateMesh_DeformMesh'` – CreateMesh + DeformMesh.  
    - `'baseFlow_import_sutherland'` – reshaping LES arrays into `baseFlow` + `sutherland_air`.  
    - `'sponge_assign'` – assigning `mesh.sponge`.  
    - `'operator_GetLin_BC_CompEnergy'` – `GetLinProblem`, `BC_Dirichlet`, `GetCompEnergyNorm`.  
    - `'resolvent'` – resolvent assembly and solve.  
    - `'plotting'` – gain and mode plots (if `verbose=true`).  
  - `timing_total` – total runtime.  
  - `mem_log` – memory after each of the above steps.

### Example 3 – `example_3_Streaky_BL.m`

- **Results (base)**: `example_3_Streaky_BL_results_base.mat`  
  - `results_StreakyBL_base.mesh` – mesh for the streaky boundary-layer configuration (if created).  
  - `results_StreakyBL_base.baseFlow` – streaky base-flow fields (if base-flow import executed).  
- **Results (temporal)**: `example_3_Streaky_BL_results_temporal.mat`  
  - `results_StreakyBL_temporal.omega` – temporal eigenvalues (if computed).  
  - `results_StreakyBL_temporal.V` – temporal eigenmodes (if computed).  
- **Results (spatial)**: `example_3_Streaky_BL_results_spatial.mat`  
  - `results_StreakyBL_spatial.alphas` – spatial eigenvalues from spatial stability analysis (if `spatialAna` is true).
- **Logs**: `example_3_Streaky_BL_logs.mat`  
  - `timing` – includes (depending on active flags):
    - `'mesh_CreateMesh_DeformMesh'` – mesh build and mapping.  
    - `'baseFlow_example3read'` – `example_3_readbaseflow` call.  
    - `'sponge_build'` – sponge construction.  
    - `'temporal_operator_GetLin_BC'` – temporal `GetLinProblem` + `BC_Dirichlet`.  
    - `'temporal_eigs'` – temporal `eigs` solve (with `opts.p` reduced to 150 in this setup to keep the computation tractable).  
    - `'temporal_plotting'` – temporal spectra and mode plots (if `verbose=true`).  
    - `'spatial_block'` – full spatial stability loop (if `spatialAna=true`).  
  - `timing_total` – total runtime.  
  - `mem_log` – memory after each listed step.

### Example 4 – `example_4_Cavity.m`

- **Results (base)**: `example_4_Cavity_results_base.mat`  
  - `results_Cavity_base.mesh` – cavity mesh after masking and deformation.  
  - `results_Cavity_base.baseFlow` – cavity base-flow fields and properties.  
- **Results (direct modes)**: `example_4_Cavity_results_direct.mat`  
  - `results_Cavity_direct.omega` – direct global eigenvalues.  
  - `results_Cavity_direct.V` – direct eigenmodes.  
- **Results (adjoint modes)**: `example_4_Cavity_results_adjoint.mat`  
  - `results_Cavity_adjoint.omegaAdj` – adjoint eigenvalues.  
  - `results_Cavity_adjoint.VAdj` – adjoint eigenmodes.
- **Logs**: `example_4_Cavity_logs.mat`  
  - `timing` – steps such as:
    - `'mesh_CreateMesh_Mask_DeformMesh'` – `CreateMesh`, `MeshMask`, and x/y mesh deformations.  
    - `'baseFlow_example4read_sutherland'` – `example_4_readbaseflow` + `sutherland_air`.  
    - `'sponge_build'` – sponge field construction.  
    - `'operator_GetLin_BC_CompEnergy'` – `GetLinProblem`, `BC_Dirichlet`, `GetCompEnergyNorm`.  
    - `'eigs_direct'` – direct `eigs` solve.  
    - `'eigs_adjoint'` – adjoint `eigs` solve.  
    - `'plotting'` – spectra and direct/adjoint mode plots (if `verbose=true`).  
  - `timing_total` – total runtime.  
  - `mem_log` – memory after each of these steps.

### Example 5 – `example_5_Paraboloid.m`

- **Results**: `example_5_Paraboloid_results.mat`  
  - `results_Paraboloid.mesh` – mesh mapped to the paraboloid geometry.  
  - `results_Paraboloid.baseFlow` – base-flow fields over the paraboloid.  
  - `results_Paraboloid.gains` – resolvent gains.  
  - `results_Paraboloid.responses` – resolvent response modes.  
  - `results_Paraboloid.forces` – resolvent forcing modes.
- **Logs**: `example_5_Paraboloid_logs.mat`  
  - `timing.total` – single scalar with total script runtime (seconds).  
  - `mem_log` – single entry:
    - `mem_log(1).step = 'end_of_script'`  
    - `mem_log(1).MB` – workspace memory in MB at the end of the run.  
  - Note: per-step timing/memory logging for example 5 is **not yet implemented**; only the total is currently recorded.

### Shared helper

- **`aux_matlab/log_step.m`**  
  - Utility function used by all examples to append timing and memory data:
    - Inputs: `timing`, `mem_log`, `stepName`, `t_start` (from `tic`).  
    - Updates:
      - `timing(end+1)` with fields `step` and `time` (seconds).  
      - `mem_log(end+1)` with fields `step` and `MB` (workspace memory in MB, measured in the caller’s workspace).

