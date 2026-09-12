# Project Analysis Summary Report (Full Version)

Generation Time: 2026-09-12 19:51:23 UTC

## Execution Timing Statistics

- **Total Execution Time**: 1216.66 seconds
- **Preprocessing Phase**: 0.06 seconds (0.0%)
- **Research Phase**: 392.40 seconds (32.3%)
- **Document Generation Phase**: 824.20 seconds (67.7%)
- **Output Phase**: 0.00 seconds (0.0%)
- **Summary Generation Time**: 0.001 seconds

## Cache Performance Statistics and Savings

### Performance Metrics
- **Cache Hit Rate**: 44.9%
- **Total Operations**: 49
- **Cache Hits**: 22 times
- **Cache Misses**: 27 times
- **Cache Writes**: 28 times

### Savings
- **Inference Time Saved**: 213.8 seconds
- **Tokens Saved**: 59544 input + 43571 output = 103115 total
- **Estimated Cost Savings**: $0.0920
- **Performance Improvement**: 44.9%
- **Efficiency Improvement Ratio**: 0.2x (saved time / actual execution time)

## Core Research Data Summary

Complete content of four types of research materials according to Prompt template data integration rules:

### System Context Research Report
Provides core objectives, user roles, and system boundary information for the project.

```json
{
  "business_value": "Lets researchers run large-scale LNSE eigenvalue and resolvent studies on HPC without MATLAB. It turns CFD mean-flow dumps into stability modes, gains, and sensitivity results for transition and flow-control analysis.",
  "confidence_score": 8.7,
  "external_systems": [
    {
      "description": "Distributed sparse matrices, KSP/PC solvers, and runtime for parallel linear algebra.",
      "interaction_type": "library dependency",
      "name": "PETSc"
    },
    {
      "description": "Eigenvalue and singular-value solvers used for temporal, spatial, adjoint, and resolvent eigenproblems.",
      "interaction_type": "library dependency",
      "name": "SLEPc"
    },
    {
      "description": "Sparse direct LU backend used as the default factorize-once solve-many path.",
      "interaction_type": "solver backend via PETSc",
      "name": "MUMPS"
    },
    {
      "description": "Process-level parallelism for distributed vectors, matrices, and case orchestration.",
      "interaction_type": "runtime dependency",
      "name": "MPI"
    },
    {
      "description": "Optional GPU sparse direct factor path shelled as a PETSc preconditioner.",
      "interaction_type": "optional solver backend",
      "name": "cuDSS"
    },
    {
      "description": "Array I/O, sparse CSC/CSR operators, and local numerical kernels before PETSc conversion.",
      "interaction_type": "library dependency",
      "name": "NumPy/SciPy"
    },
    {
      "description": "On-disk mean-flow fields as npy directories or npz archives (rho, u, v, w, T, mesh, sponge).",
      "interaction_type": "file input",
      "name": "CFD baseflow data stores"
    },
    {
      "description": "Case, mesh, eigen, resolvent, and solver policy files merged with defaults and schema checks.",
      "interaction_type": "file input",
      "name": "YAML case configuration"
    }
  ],
  "project_description": "Python toolkit for linear stability and resolvent analysis of compressible flows. It builds linearized Navier-Stokes operators on structured meshes with SBP finite differences, then solves temporal, spatial, adjoint, and resolvent problems with PETSc/SLEPc. It is a Python port of LinStab2D MATLAB workflows with MPI-distributed case orchestration.",
  "project_name": "LinStabPy",
  "project_type": "CLITool",
  "system_boundary": {
    "excluded_components": [
      "Nonlinear CFD solvers that produce the mean flow",
      "GUI or interactive visualization front ends",
      "Unstructured-mesh or fully 3D volume discretization frameworks",
      "General-purpose non-LNSE multiphysics simulation",
      "Cloud deployment, auth, or multi-tenant service layers"
    ],
    "included_components": [
      "Baseflow loading and preprocess",
      "Structured mesh with SBP operators and filters",
      "LNSE coefficient generation and operator assembly",
      "Boundary-condition recipes (Dirichlet/Neumann)",
      "Temporal, spatial, and adjoint eigenvalue solvers",
      "Resolvent operator, gain extraction, and beta/frequency maps",
      "Sparse factorization layer (MUMPS/LU/cuDSS)",
      "Solver routing with anchors, recycling, and stable sweeps",
      "CLI/driver case orchestration, config validation, I/O, and provenance"
    ],
    "scope": "In-scope work starts from loaded mean-flow and mesh geometry and ends at stability/resolvent results, metrics, and payloads. The system assembles LNSE operators, enforces BCs, factorizes shifted systems, and runs eigen/resolvent workflows. It does not generate nonlinear CFD baseflows or provide end-user visualization products."
  },
  "target_users": [
    {
      "description": "Scientists studying hydrodynamic stability, transition, and receptivity in compressible flows.",
      "name": "Fluid dynamics researchers",
      "needs": [
        "Temporal and spatial eigenvalue spectra",
        "Adjoint modes and structural sensitivity",
        "Resolvent gain maps over frequency and spanwise wavenumber",
        "Reproducible YAML-driven case runs on MPI clusters"
      ]
    },
    {
      "description": "Practitioners who already have mean-flow fields and need linear analysis of shear layers, boundary layers, or jets.",
      "name": "CFD and aerospace engineers",
      "needs": [
        "Ingest baseflow from npy/npz dumps",
        "Apply Dirichlet/Neumann BC recipes on LNSE operators",
        "Direct sparse solves with MUMPS or GPU factors",
        "Cost-aware frequency sweeps with factor reuse"
      ]
    },
    {
      "description": "Developers extending or embedding LNSE analysis in larger openfluids pipelines.",
      "name": "Scientific software developers",
      "needs": [
        "Importable library API without PETSc init on import",
        "Typed config and schema validation",
        "SciPy and PETSc backend bridges",
        "Provenance and metrics for solver routing"
      ]
    }
  ]
}
```

### Domain Modules Research Report
Provides high-level domain division, module relationships, and core business process information.

```json
{
  "architecture_summary": "LinStabPy is a Python CLI/library toolkit that ports LinStab2D-style compressible LNSE workflows to MPI-enabled HPC. Architecturally it is a staged scientific pipeline: YAML-driven case orchestration loads mean-flow data onto structured SBP meshes, assembles BC-enforced linearized Navier-Stokes operators, then runs temporal/spatial/adjoint eigenproblems and resolvent gain analyses through PETSc/SLEPc with a factorize-once (MUMPS/LU/optional cuDSS) linear-algebra core. A dedicated solver-router layer optimizes large beta–frequency campaigns via anchors, preconditioner reuse, and residual gates. Domain boundaries align cleanly with packages (driver, config, baseflow, mesh, operators+boundary_conditions, stability, resolvent, solvers/router, linalg, utils); boundary conditions are functionally part of operator assembly though physically separate in code. External CFD, visualization, and unstructured 3D frameworks remain out of scope.",
  "business_flows": [
    {
      "description": "Loads mean flow and mesh, assembles the temporal LNSE operator with BCs, factorizes shifted systems, and computes eigenvalues/eigenvectors (and optionally adjoint/sensitivity) for hydrodynamic stability.",
      "entry_point": "src/linstabpy/driver/run.py or stability eigenvalue API via package import",
      "importance": 9.8,
      "involved_domains_count": 8,
      "name": "Temporal Eigenvalue Stability Analysis",
      "steps": [
        {
          "code_entry_point": "src/linstabpy/driver/run.py",
          "domain_module": "Case Orchestration",
          "operation": "Initialize PETSc/MPI, resolve case paths, and start staged orchestration",
          "step": 1,
          "sub_module": "Runtime Pipeline Driver"
        },
        {
          "code_entry_point": "src/linstabpy/config/typed.py",
          "domain_module": "Configuration Management",
          "operation": "Merge YAML, environment, and defaults; validate eigen/mesh/MPI policy",
          "step": 2,
          "sub_module": "Typed Config Model"
        },
        {
          "code_entry_point": "src/linstabpy/baseflow/grid_fields.py",
          "domain_module": "Baseflow Management",
          "operation": "Load CFD mean-flow npy/npz fields into BaseFlow and apply preprocess/viscosity",
          "step": 3,
          "sub_module": "Grid Field I/O"
        },
        {
          "code_entry_point": "src/linstabpy/mesh/mesh.py",
          "domain_module": "Mesh Discretization",
          "operation": "Build structured mesh, SBP operators, filters, and runtime state",
          "step": 4,
          "sub_module": "Mesh Core"
        },
        {
          "code_entry_point": "src/linstabpy/operators/coefficients.py",
          "domain_module": "LNSE Operator Assembly",
          "operation": "Compute LNSE coefficients from baseflow for Cartesian or cylindrical flow",
          "step": 5,
          "sub_module": "Physical Coefficients"
        },
        {
          "code_entry_point": "src/linstabpy/operators/assembly.py",
          "domain_module": "LNSE Operator Assembly",
          "operation": "Assemble sparse L0 operator and apply BC recipes",
          "step": 6,
          "sub_module": "Temporal Operator Assembly"
        },
        {
          "code_entry_point": "src/linstabpy/linalg/factorize.py",
          "domain_module": "Linear Algebra Infrastructure",
          "operation": "Factorize shifted LNSE operator for shift-invert eigen path",
          "step": 7,
          "sub_module": "Factorization Layer"
        },
        {
          "code_entry_point": "src/linstabpy/stability/_eigenvalues.py",
          "domain_module": "Stability Analysis",
          "operation": "Run SLEPc temporal/adjoint eigen solve, validate modes, return spectra and metadata",
          "step": 8,
          "sub_module": "Stability Eigenvalue Services"
        },
        {
          "code_entry_point": "src/linstabpy/driver/payload.py",
          "domain_module": "Case Orchestration",
          "operation": "Emit eigenvalues, eigenvectors, metrics, and provenance payload",
          "step": 9,
          "sub_module": "CLI and I/O Payload"
        }
      ]
    },
    {
      "description": "Constructs the resolvent operator from L0 at target frequencies, solves weighted optimal forcing/response problems, and reports gains and mode shapes for amplifier/receptivity analysis.",
      "entry_point": "src/linstabpy/driver/run.py or resolvent.analysis entry",
      "importance": 9.6,
      "involved_domains_count": 8,
      "name": "Resolvent Gain Analysis",
      "steps": [
        {
          "code_entry_point": "src/linstabpy/driver/run.py",
          "domain_module": "Case Orchestration",
          "operation": "Initialize runtime and select resolvent case stage",
          "step": 1,
          "sub_module": "Runtime Pipeline Driver"
        },
        {
          "code_entry_point": "src/linstabpy/config/typed.py",
          "domain_module": "Configuration Management",
          "operation": "Resolve resolvent frequency, weights, and solver backend policy",
          "step": 2,
          "sub_module": "Typed Config Model"
        },
        {
          "code_entry_point": "src/linstabpy/baseflow/baseflow.py",
          "domain_module": "Baseflow Management",
          "operation": "Provide mesh-aligned mean flow for linearization",
          "step": 3,
          "sub_module": "BaseFlow Model"
        },
        {
          "code_entry_point": "src/linstabpy/mesh/sbp.py",
          "domain_module": "Mesh Discretization",
          "operation": "Supply differentiation and norm operators for discrete L0 and energy weights",
          "step": 4,
          "sub_module": "SBP Operators"
        },
        {
          "code_entry_point": "src/linstabpy/operators/assembly.py",
          "domain_module": "LNSE Operator Assembly",
          "operation": "Assemble L0 with BCs and energy-norm related maps",
          "step": 5,
          "sub_module": "Temporal Operator Assembly"
        },
        {
          "code_entry_point": "src/linstabpy/linalg/factorize.py",
          "domain_module": "Linear Algebra Infrastructure",
          "operation": "Factor (L0 - i omega I) for repeated resolvent applies",
          "step": 6,
          "sub_module": "Factorization Layer"
        },
        {
          "code_entry_point": "src/linstabpy/resolvent/operator.py",
          "domain_module": "Resolvent Analysis",
          "operation": "Apply H/H^H and build weighted normal operator shell",
          "step": 7,
          "sub_module": "Resolvent Operator and Gain"
        },
        {
          "code_entry_point": "src/linstabpy/resolvent/analysis.py",
          "domain_module": "Resolvent Analysis",
          "operation": "Extract optimal forcing/response pairs and gains via SLEPc; validate residuals",
          "step": 8,
          "sub_module": "Resolvent Analysis Driver"
        },
        {
          "code_entry_point": "src/linstabpy/driver/payload.py",
          "domain_module": "Case Orchestration",
          "operation": "Package gains, modes, and solver metrics for output",
          "step": 9,
          "sub_module": "CLI and I/O Payload"
        }
      ]
    },
    {
      "description": "Parametric HPC campaign: assemble LNSE per spanwise wavenumber beta, mark frequency bands, walk lines with anchor factors and residual-gated reuse, and record leading gains across the map.",
      "entry_point": "src/linstabpy/solvers/router/resolvent_map.py",
      "importance": 9.4,
      "involved_domains_count": 7,
      "name": "Beta-Frequency Resolvent Gain Map",
      "steps": [
        {
          "code_entry_point": "src/linstabpy/driver/run.py",
          "domain_module": "Case Orchestration",
          "operation": "Launch distributed map campaign under case config",
          "step": 1,
          "sub_module": "Runtime Pipeline Driver"
        },
        {
          "code_entry_point": "src/linstabpy/config/schema.py",
          "domain_module": "Configuration Management",
          "operation": "Validate resolvent map, router, and resource policy sections",
          "step": 2,
          "sub_module": "Schema Validation"
        },
        {
          "code_entry_point": "src/linstabpy/operators/assembly.py",
          "domain_module": "LNSE Operator Assembly",
          "operation": "Assemble one LNSE operator per beta including BCs",
          "step": 3,
          "sub_module": "Temporal Operator Assembly"
        },
        {
          "code_entry_point": "src/linstabpy/solvers/router/resolvent_map.py",
          "domain_module": "Solver Routing and Sweep Optimization",
          "operation": "Mark frequency bands and schedule line walks per beta",
          "step": 4,
          "sub_module": "Resolvent Map Orchestration"
        },
        {
          "code_entry_point": "src/linstabpy/solvers/router/map_walk.py",
          "domain_module": "Solver Routing and Sweep Optimization",
          "operation": "Factor anchors, reuse as preconditioners, residual-gate refactor, optional stable sweep",
          "step": 5,
          "sub_module": "Anchor and Line Walk Engines"
        },
        {
          "code_entry_point": "src/linstabpy/linalg/backends.py",
          "domain_module": "Linear Algebra Infrastructure",
          "operation": "Select and probe MUMPS/LU/cuDSS backends under RAM/cost policy",
          "step": 6,
          "sub_module": "Backend Registry"
        },
        {
          "code_entry_point": "src/linstabpy/resolvent/analysis.py",
          "domain_module": "Resolvent Analysis",
          "operation": "Compute leading resolvent gain at each scheduled frequency",
          "step": 7,
          "sub_module": "Resolvent Analysis Driver"
        },
        {
          "code_entry_point": "src/linstabpy/driver/payload.py",
          "domain_module": "Case Orchestration",
          "operation": "Record map results, cost samples, and certification metadata",
          "step": 8,
          "sub_module": "CLI and I/O Payload"
        }
      ]
    },
    {
      "description": "Extends prepared LNSE coefficients into spatial block operators and solves generalized spatial eigenvalue problems with SLEPc for streamwise wavenumber/growth analysis.",
      "entry_point": "src/linstabpy/solvers/spatial_eig.py or stability spatial API",
      "importance": 8.8,
      "involved_domains_count": 6,
      "name": "Spatial Stability Eigenproblem",
      "steps": [
        {
          "code_entry_point": "src/linstabpy/config/typed.py",
          "domain_module": "Configuration Management",
          "operation": "Resolve solver.spatial policy with fallback to solver.eigen",
          "step": 1,
          "sub_module": "Typed Config Model"
        },
        {
          "code_entry_point": "src/linstabpy/baseflow/baseflow.py",
          "domain_module": "Baseflow Management",
          "operation": "Supply mean-flow fields for spatial linearization",
          "step": 2,
          "sub_module": "BaseFlow Model"
        },
        {
          "code_entry_point": "src/linstabpy/mesh/mesh.py",
          "domain_module": "Mesh Discretization",
          "operation": "Provide mesh indexing and SBP operators for spatial discretization",
          "step": 3,
          "sub_module": "Mesh Core"
        },
        {
          "code_entry_point": "src/linstabpy/operators/spatial.py",
          "domain_module": "LNSE Operator Assembly",
          "operation": "Build spatial stability block operators for SciPy/PETSc",
          "step": 4,
          "sub_module": "Spatial Operator Assembly"
        },
        {
          "code_entry_point": "src/linstabpy/linalg/factorize.py",
          "domain_module": "Linear Algebra Infrastructure",
          "operation": "Configure direct shift-invert factor backend for spatial EPS",
          "step": 5,
          "sub_module": "Factorization Layer"
        },
        {
          "code_entry_point": "src/linstabpy/solvers/spatial_eig.py",
          "domain_module": "Stability Analysis",
          "operation": "Run generalized SLEPc spatial eigen solve and return modes",
          "step": 6,
          "sub_module": "Eigen Solver Backends"
        }
      ]
    },
    {
      "description": "Standalone preparation path from on-disk CFD dumps through BaseFlow preprocess, mesh/SBP setup, coefficient generation, and BC-ready L0 assembly prior to any eigen or resolvent solve.",
      "entry_point": "src/linstabpy/driver/run.py preflight/setup stages",
      "importance": 8.5,
      "involved_domains_count": 5,
      "name": "Baseflow Ingest and Operator Prep",
      "steps": [
        {
          "code_entry_point": "src/linstabpy/config/schema.py",
          "domain_module": "Configuration Management",
          "operation": "Validate case dictionary structure and FD orders",
          "step": 1,
          "sub_module": "Schema Validation"
        },
        {
          "code_entry_point": "src/linstabpy/baseflow/grid_fields.py",
          "domain_module": "Baseflow Management",
          "operation": "Ingest npy/npz primitives and optional sponge fields",
          "step": 2,
          "sub_module": "Grid Field I/O"
        },
        {
          "code_entry_point": "src/linstabpy/baseflow/preprocess.py",
          "domain_module": "Baseflow Management",
          "operation": "Apply shock cleanup, sponge, and viscosity transport fill",
          "step": 3,
          "sub_module": "Preprocess and Transport"
        },
        {
          "code_entry_point": "src/linstabpy/mesh/mesh.py",
          "domain_module": "Mesh Discretization",
          "operation": "Create/deform mesh and attach SBP operators and filters",
          "step": 4,
          "sub_module": "Mesh Core"
        },
        {
          "code_entry_point": "src/linstabpy/boundary_conditions/recipe.py",
          "domain_module": "LNSE Operator Assembly",
          "operation": "Apply declarative Dirichlet/Neumann recipes and obtain free-DOF sets",
          "step": 5,
          "sub_module": "Boundary Condition Recipes"
        },
        {
          "code_entry_point": "src/linstabpy/operators/assembly.py",
          "domain_module": "LNSE Operator Assembly",
          "operation": "Finalize sparse L0 for downstream stability or resolvent stages",
          "step": 6,
          "sub_module": "Temporal Operator Assembly"
        }
      ]
    }
  ],
  "confidence_score": 9.1,
  "domain_modules": [
    {
      "code_paths": [
        "src/linstabpy/driver",
        "src/linstabpy/__init__.py"
      ],
      "complexity": 8.5,
      "description": "Owns end-to-end LNSE case execution: PETSc/MPI initialization, config resolution, staged workflow coordination, payload assembly, result emission, and CLI entry. Primary business hub connecting baseflow, operators, stability, and resolvent stages for reproducible HPC runs.",
      "domain_type": "Core Business Domain",
      "importance": 9.7,
      "name": "Case Orchestration",
      "sub_modules": [
        {
          "code_paths": [
            "src/linstabpy/driver/run.py",
            "src/linstabpy/driver/preflight.py",
            "src/linstabpy/driver/__init__.py"
          ],
          "description": "Core shared case-pipeline that initializes PETSc/MPI, coordinates distributed vectors, logging/heartbeats, and executes stability/resolvent workflows end-to-end.",
          "importance": 9.8,
          "key_functions": [
            "PETSc/MPI initialization",
            "config resolution",
            "distributed case orchestration",
            "staged logging and heartbeats"
          ],
          "name": "Runtime Pipeline Driver"
        },
        {
          "code_paths": [
            "src/linstabpy/driver/cli.py",
            "src/linstabpy/driver/io.py",
            "src/linstabpy/driver/payload.py"
          ],
          "description": "Command-line entry, case I/O helpers, and result/payload packaging for downstream consumption and provenance.",
          "importance": 8.5,
          "key_functions": [
            "CLI invocation",
            "case input loading",
            "result payload assembly",
            "output emission"
          ],
          "name": "CLI and I/O Payload"
        },
        {
          "code_paths": [
            "src/linstabpy/__init__.py"
          ],
          "description": "Lightweight package entry exposing stability, adjoint, sensitivity, and resolvent workflows via lazy loading without starting PETSc/MPI on import.",
          "importance": 9.0,
          "key_functions": [
            "lazy public symbol export",
            "version and API discovery",
            "import-safe library surface"
          ],
          "name": "Package Public API"
        }
      ]
    },
    {
      "code_paths": [
        "src/linstabpy/config"
      ],
      "complexity": 6.5,
      "description": "Resolves YAML cases, environment overrides, and defaults into typed solver policy; validates schema before runs so eigen, resolvent, mesh, and MPI settings are consistent and reproducible.",
      "domain_type": "Supporting Domain",
      "importance": 8.8,
      "name": "Configuration Management",
      "sub_modules": [
        {
          "code_paths": [
            "src/linstabpy/config/typed.py",
            "src/linstabpy/config/defaults.py",
            "src/linstabpy/config/__init__.py"
          ],
          "description": "Dataclass seam for linear-solve, resolvent, eigen, mesh, and MPI subsections enabling typed access without behavior change.",
          "importance": 9.5,
          "key_functions": [
            "YAML/env/default merge",
            "typed solver policy objects",
            "gradual migration to typed access"
          ],
          "name": "Typed Config Model"
        },
        {
          "code_paths": [
            "src/linstabpy/config/schema.py",
            "src/linstabpy/config/validation_cases.yaml"
          ],
          "description": "Lenient validation of merged case dictionaries with allowed keys and finite-difference order checks.",
          "importance": 9.0,
          "key_functions": [
            "merged config validation",
            "unknown-key rejection",
            "FD-order checks",
            "ConfigValidationError"
          ],
          "name": "Schema Validation"
        }
      ]
    },
    {
      "code_paths": [
        "src/linstabpy/baseflow"
      ],
      "complexity": 7.0,
      "description": "Ingests CFD mean-flow dumps into mesh-aligned BaseFlow state, applies preprocess/sponge/viscosity helpers, and supplies primitive and derived fields for LNSE linearization.",
      "domain_type": "Core Business Domain",
      "importance": 9.3,
      "name": "Baseflow Management",
      "sub_modules": [
        {
          "code_paths": [
            "src/linstabpy/baseflow/baseflow.py",
            "src/linstabpy/baseflow/__init__.py"
          ],
          "description": "Central dataclass holding mesh-aligned flow fields and parameters used throughout the stability pipeline.",
          "importance": 9.5,
          "key_functions": [
            "BaseFlow state representation",
            "(ny,nx) field contracts",
            "parameter carriage for LNSE"
          ],
          "name": "BaseFlow Model"
        },
        {
          "code_paths": [
            "src/linstabpy/baseflow/grid_fields.py"
          ],
          "description": "Loads rectilinear primitives from npy directories or npz archives into BaseFlow-ready dictionaries.",
          "importance": 8.8,
          "key_functions": [
            "npy/npz mean-flow load",
            "field name validation",
            "preprocess hook after load"
          ],
          "name": "Grid Field I/O"
        },
        {
          "code_paths": [
            "src/linstabpy/baseflow/preprocess.py",
            "src/linstabpy/baseflow/sponge.py",
            "src/linstabpy/baseflow/shock_cleanup.py",
            "src/linstabpy/baseflow/viscosity.py"
          ],
          "description": "Sponge, shock cleanup, viscosity, and related preprocess steps that complete mean-flow readiness for operator assembly.",
          "importance": 8.2,
          "key_functions": [
            "sponge application",
            "shock cleanup",
            "viscosity coefficient fill",
            "baseflow preprocess pipeline"
          ],
          "name": "Preprocess and Transport"
        }
      ]
    },
    {
      "code_paths": [
        "src/linstabpy/mesh"
      ],
      "complexity": 8.8,
      "description": "Structured-mesh geometry, SBP finite-difference operators, filters, deformation, masking, and runtime state, foundation for spatial discretization of LNSE problems.",
      "domain_type": "Core Business Domain",
      "importance": 9.6,
      "name": "Mesh Discretization",
      "sub_modules": [
        {
          "code_paths": [
            "src/linstabpy/mesh/mesh.py",
            "src/linstabpy/mesh/__init__.py"
          ],
          "description": "Primary Mesh object with geometry, operators, and runtime-state views; creation, deformation, pole-line handling, and masking.",
          "importance": 9.8,
          "key_functions": [
            "mesh creation and deformation",
            "pole-line handling",
            "masking",
            "operator attachment",
            "geometry/runtime views"
          ],
          "name": "Mesh Core"
        },
        {
          "code_paths": [
            "src/linstabpy/mesh/sbp.py",
            "src/linstabpy/mesh/sbp_generator.py",
            "src/linstabpy/mesh/fdweights.py"
          ],
          "description": "Summation-by-parts stencils and sparse matrices (orders 2–8) including pole treatments for axisymmetric grids.",
          "importance": 9.3,
          "key_functions": [
            "SBP differentiation matrices",
            "norm/dissipation operators",
            "Mohseni–Colonius pole matrices",
            "FD weight construction"
          ],
          "name": "SBP Operators"
        },
        {
          "code_paths": [
            "src/linstabpy/mesh/filters.py"
          ],
          "description": "Filtering utilities attached to mesh runtime for numerical stabilization of discrete operators.",
          "importance": 7.5,
          "key_functions": [
            "filter construction",
            "mesh-integrated filtering"
          ],
          "name": "Mesh Filters"
        }
      ]
    },
    {
      "code_paths": [
        "src/linstabpy/operators",
        "src/linstabpy/boundary_conditions"
      ],
      "complexity": 9.2,
      "description": "Builds linearized Navier-Stokes operators from baseflow coefficients and FD matrices, applies boundary-condition recipes, and produces temporal/spatial systems for SciPy and PETSc backends.",
      "domain_type": "Core Business Domain",
      "importance": 9.7,
      "name": "LNSE Operator Assembly",
      "sub_modules": [
        {
          "code_paths": [
            "src/linstabpy/operators/assembly.py",
            "src/linstabpy/operators/__init__.py"
          ],
          "description": "Main LNSE assembly pipeline producing the large sparse L0 system for state q=[rho,u,v,w,T] with sponge and backend finalization.",
          "importance": 9.8,
          "key_functions": [
            "temporal LNSE assembly",
            "SciPy CSC and PETSc Mat paths",
            "sponge application",
            "request-driven backend selection"
          ],
          "name": "Temporal Operator Assembly"
        },
        {
          "code_paths": [
            "src/linstabpy/operators/coefficients.py",
            "src/linstabpy/operators/metrics.py",
            "src/linstabpy/operators/energy_norm.py"
          ],
          "description": "Computes LNSE coefficient matrices for Cartesian and cylindrical compressible flows from baseflow fields.",
          "importance": 9.3,
          "key_functions": [
            "coefficient generation",
            "coordinate-system transforms",
            "energy-norm weights",
            "metrics for operators"
          ],
          "name": "Physical Coefficients"
        },
        {
          "code_paths": [
            "src/linstabpy/operators/spatial.py"
          ],
          "description": "Assembles spatial stability eigenproblem operators reusing temporal preparation helpers.",
          "importance": 8.8,
          "key_functions": [
            "spatial eigenproblem assembly",
            "block operator construction",
            "PETSc block-matrix paths"
          ],
          "name": "Spatial Operator Assembly"
        },
        {
          "code_paths": [
            "src/linstabpy/boundary_conditions/recipe.py",
            "src/linstabpy/boundary_conditions/dirichlet.py",
            "src/linstabpy/boundary_conditions/neumann.py",
            "src/linstabpy/boundary_conditions/__init__.py"
          ],
          "description": "Declarative Dirichlet/Neumann BC rules sequenced onto LNSE operators with free-DOF index sets.",
          "importance": 9.2,
          "key_functions": [
            "BCRule/BCRecipe application",
            "Dirichlet row/column enforcement",
            "Neumann conditions",
            "free-DOF mask construction"
          ],
          "name": "Boundary Condition Recipes"
        }
      ]
    },
    {
      "code_paths": [
        "src/linstabpy/stability",
        "src/linstabpy/solvers/eigenvalue.py",
        "src/linstabpy/solvers/spatial_eig.py",
        "src/linstabpy/solvers/_eps_helpers.py",
        "src/linstabpy/solvers/petsc_utils.py",
        "src/linstabpy/solvers/__init__.py"
      ],
      "complexity": 9.0,
      "description": "Temporal, spatial, and adjoint eigenvalue solves plus structural sensitivity, core scientific product for hydrodynamic stability and transition studies.",
      "domain_type": "Core Business Domain",
      "importance": 9.6,
      "name": "Stability Analysis",
      "sub_modules": [
        {
          "code_paths": [
            "src/linstabpy/stability/_eigenvalues.py",
            "src/linstabpy/stability/__init__.py",
            "src/linstabpy/stability/acceptance.py"
          ],
          "description": "Primary entry points for temporal, adjoint, and spatial eigen solves with SLEPc/PETSc, validation, and metadata return.",
          "importance": 9.5,
          "key_functions": [
            "temporal eigenanalysis",
            "adjoint eigenanalysis",
            "spatial eigen solves",
            "mode/weight validation",
            "SLEPc shift-invert drive"
          ],
          "name": "Stability Eigenvalue Services"
        },
        {
          "code_paths": [
            "src/linstabpy/solvers/eigenvalue.py",
            "src/linstabpy/solvers/spatial_eig.py",
            "src/linstabpy/solvers/_eps_helpers.py",
            "src/linstabpy/solvers/petsc_utils.py",
            "src/linstabpy/solvers/__init__.py"
          ],
          "description": "SLEPc EPS implementations for temporal/adjoint and spatial problems with factor-backend selection and residual handling.",
          "importance": 9.5,
          "key_functions": [
            "shift-invert EPS",
            "MUMPS/LU backend selection",
            "residual reordering",
            "spatial generalized eigenproblems"
          ],
          "name": "Eigen Solver Backends"
        },
        {
          "code_paths": [
            "src/linstabpy/stability/sensitivity.py"
          ],
          "description": "Sensitivity analysis building on direct/adjoint modes for flow-control and receptivity insight.",
          "importance": 8.0,
          "key_functions": [
            "structural sensitivity computation",
            "direct-adjoint mode combination"
          ],
          "name": "Structural Sensitivity"
        }
      ]
    },
    {
      "code_paths": [
        "src/linstabpy/resolvent"
      ],
      "complexity": 8.8,
      "description": "Resolvent operator construction, optimal forcing/response pairs, weighted gain extraction, and related forcing models for receptivity and amplifier analysis.",
      "domain_type": "Core Business Domain",
      "importance": 9.5,
      "name": "Resolvent Analysis",
      "sub_modules": [
        {
          "code_paths": [
            "src/linstabpy/resolvent/analysis.py",
            "src/linstabpy/resolvent/__init__.py"
          ],
          "description": "Computes optimal forcing/response pairs and gains via H = C inv(L0 - i omega I) B using SLEPc on weighted normal operators.",
          "importance": 9.5,
          "key_functions": [
            "resolvent analysis workflow",
            "gain and mode extraction",
            "residual validation",
            "MATLAB-aligned path"
          ],
          "name": "Resolvent Analysis Driver"
        },
        {
          "code_paths": [
            "src/linstabpy/resolvent/operator.py",
            "src/linstabpy/resolvent/gain.py",
            "src/linstabpy/resolvent/conservative_forcing.py"
          ],
          "description": "Forward/adjoint actions of H and MatShell weighted normal operator for SLEPc gain eigenproblems.",
          "importance": 9.0,
          "key_functions": [
            "H and H^H matvecs",
            "WeightedNormalOperator shell",
            "input/output maps B and C",
            "conservative forcing options"
          ],
          "name": "Resolvent Operator and Gain"
        }
      ]
    },
    {
      "code_paths": [
        "src/linstabpy/solvers/router"
      ],
      "complexity": 9.0,
      "description": "Cost-aware beta/frequency resolvent map orchestration with anchor factorization, preconditioner reuse, residual gates, and stable-regime sweeps for large parametric studies.",
      "domain_type": "Core Business Domain",
      "importance": 9.2,
      "name": "Solver Routing and Sweep Optimization",
      "sub_modules": [
        {
          "code_paths": [
            "src/linstabpy/solvers/router/resolvent_map.py",
            "src/linstabpy/solvers/router/__init__.py",
            "src/linstabpy/solvers/router/map_record.py",
            "src/linstabpy/solvers/router/band_marking.py"
          ],
          "description": "Turnkey beta/frequency gain-map entry assembling one LNSE operator per beta and walking frequency lines with factor reuse.",
          "importance": 9.5,
          "key_functions": [
            "beta/frequency map drive",
            "per-beta operator assembly",
            "frequency band marking",
            "gain-map recording"
          ],
          "name": "Resolvent Map Orchestration"
        },
        {
          "code_paths": [
            "src/linstabpy/solvers/router/map_walk.py",
            "src/linstabpy/solvers/router/stable_sweep.py",
            "src/linstabpy/solvers/router/anchors.py",
            "src/linstabpy/solvers/router/recycling.py",
            "src/linstabpy/solvers/router/break_even.py",
            "src/linstabpy/solvers/router/ram_autoselect.py",
            "src/linstabpy/solvers/router/certification.py"
          ],
          "description": "Measured MUMPS anchors, residual-gated frequency walks, stable sweeps, recycling, and RAM-aware backend selection.",
          "importance": 9.3,
          "key_functions": [
            "anchor factorization",
            "preconditioned frequency sweep",
            "residual gate and refactor",
            "cost break-even logic",
            "RAM autoselect",
            "result certification"
          ],
          "name": "Anchor and Line Walk Engines"
        }
      ]
    },
    {
      "code_paths": [
        "src/linstabpy/linalg",
        "src/linstabpy/mumps_options.py"
      ],
      "complexity": 8.5,
      "description": "Reusable PETSc factorization, backend registry (MUMPS/LU/cuDSS), and shifted-system helpers enabling factorize-once solve-many paths for eigen and resolvent workflows.",
      "domain_type": "Infrastructure Domain",
      "importance": 9.4,
      "name": "Linear Algebra Infrastructure",
      "sub_modules": [
        {
          "code_paths": [
            "src/linstabpy/linalg/factorize.py",
            "src/linstabpy/linalg/shifted.py",
            "src/linstabpy/linalg/cudss_factor.py",
            "src/linstabpy/linalg/__init__.py"
          ],
          "description": "Direct-by-default factorize-once solve-many interfaces including transpose/block solve, refactorize, and optional cuDSS PC.",
          "importance": 9.5,
          "key_functions": [
            "KSPPREONLY+PCLU+MUMPS path",
            "solve and transpose solve",
            "block solve",
            "refactorize/destroy",
            "cuDSS shell PC"
          ],
          "name": "Factorization Layer"
        },
        {
          "code_paths": [
            "src/linstabpy/linalg/backends.py",
            "src/linstabpy/mumps_options.py"
          ],
          "description": "Sparse-direct factor backend probing and typed options for MUMPS ordering, BLR, precision, and availability.",
          "importance": 9.0,
          "key_functions": [
            "backend name normalization",
            "availability probing",
            "MUMPS option mapping",
            "precision policy"
          ],
          "name": "Backend Registry"
        }
      ]
    },
    {
      "code_paths": [
        "src/linstabpy/utils",
        "src/linstabpy/metrics_schema.py"
      ],
      "complexity": 7.0,
      "description": "Cross-cutting PETSc/SciPy bridges, solver config helpers, YAML I/O, cost metrics, provenance, and Floquet utilities shared by mesh and solver layers.",
      "domain_type": "Infrastructure Domain",
      "importance": 8.5,
      "name": "Shared Scientific Utilities",
      "sub_modules": [
        {
          "code_paths": [
            "src/linstabpy/utils/petsc_core.py",
            "src/linstabpy/utils/petsc_init.py",
            "src/linstabpy/utils/__init__.py"
          ],
          "description": "SciPy-to-PETSc conversion and matrix/vector utilities for local and distributed workflows.",
          "importance": 9.5,
          "key_functions": [
            "sparse SciPy to PETSc conversion",
            "distributed vector helpers",
            "complex-mode policy",
            "PETSc init ordering"
          ],
          "name": "PETSc Core Bridge"
        },
        {
          "code_paths": [
            "src/linstabpy/utils/_solver_cfg.py",
            "src/linstabpy/utils/yaml_io.py",
            "src/linstabpy/utils/cost_metrics.py",
            "src/linstabpy/utils/provenance.py",
            "src/linstabpy/utils/floquet.py",
            "src/linstabpy/metrics_schema.py"
          ],
          "description": "Solver config resolution mirrors, YAML helpers, cost metrics, provenance, and Floquet support.",
          "importance": 8.0,
          "key_functions": [
            "solver cfg resolution",
            "YAML load/dump",
            "cost metric recording",
            "provenance tracking",
            "Floquet helpers"
          ],
          "name": "Config I/O Metrics and Provenance"
        }
      ]
    }
  ],
  "domain_relations": [
    {
      "description": "Driver resolves and validates typed YAML/env case policy before staging mesh, baseflow, and solver work.",
      "from_domain": "Case Orchestration",
      "relation_type": "Configuration Dependency",
      "strength": 9.0,
      "to_domain": "Configuration Management"
    },
    {
      "description": "Orchestration loads and preprocesses mean-flow fields that seed all downstream LNSE analyses.",
      "from_domain": "Case Orchestration",
      "relation_type": "Data Dependency",
      "strength": 9.0,
      "to_domain": "Baseflow Management"
    },
    {
      "description": "Driver invokes temporal, spatial, and adjoint eigenvalue workflows as primary case stages.",
      "from_domain": "Case Orchestration",
      "relation_type": "Service Call",
      "strength": 9.5,
      "to_domain": "Stability Analysis"
    },
    {
      "description": "Driver invokes resolvent analysis and coordinates with router-driven gain-map campaigns.",
      "from_domain": "Case Orchestration",
      "relation_type": "Service Call",
      "strength": 9.5,
      "to_domain": "Resolvent Analysis"
    },
    {
      "description": "Driver delegates beta/frequency map orchestration and cost-aware sweeps to the router package.",
      "from_domain": "Case Orchestration",
      "relation_type": "Service Call",
      "strength": 9.0,
      "to_domain": "Solver Routing and Sweep Optimization"
    },
    {
      "description": "Case pipeline depends on assembled L0/spatial operators as the mathematical core of each solve stage.",
      "from_domain": "Case Orchestration",
      "relation_type": "Module Dependency",
      "strength": 8.5,
      "to_domain": "LNSE Operator Assembly"
    },
    {
      "description": "Orchestration and solver stages rely on shared factorization and PETSc linear-algebra services.",
      "from_domain": "Case Orchestration",
      "relation_type": "Module Dependency",
      "strength": 8.5,
      "to_domain": "Linear Algebra Infrastructure"
    },
    {
      "description": "Operator assembly composes SBP differentiation, norms, and mesh index sets into sparse LNSE matrices.",
      "from_domain": "LNSE Operator Assembly",
      "relation_type": "Composition",
      "strength": 9.5,
      "to_domain": "Mesh Discretization"
    },
    {
      "description": "Physical coefficients and linearization are driven entirely by mesh-aligned BaseFlow primitives and transport fields.",
      "from_domain": "LNSE Operator Assembly",
      "relation_type": "Data Dependency",
      "strength": 9.5,
      "to_domain": "Baseflow Management"
    },
    {
      "description": "Assembled operators are converted and consumed as PETSc/SciPy matrices for subsequent factor and eigen stages.",
      "from_domain": "LNSE Operator Assembly",
      "relation_type": "Module Dependency",
      "strength": 7.5,
      "to_domain": "Linear Algebra Infrastructure"
    },
    {
      "description": "Shift-invert eigen solves factorize and repeatedly solve shifted LNSE systems via the factorization layer.",
      "from_domain": "Stability Analysis",
      "relation_type": "Service Call",
      "strength": 9.5,
      "to_domain": "Linear Algebra Infrastructure"
    },
    {
      "description": "Eigenproblems operate on assembled temporal or spatial LNSE operators with BCs already enforced.",
      "from_domain": "Stability Analysis",
      "relation_type": "Data Dependency",
      "strength": 9.0,
      "to_domain": "LNSE Operator Assembly"
    },
    {
      "description": "Resolvent H actions require repeated solves against (L0 - i omega I) using factorize-once backends.",
      "from_domain": "Resolvent Analysis",
      "relation_type": "Service Call",
      "strength": 9.5,
      "to_domain": "Linear Algebra Infrastructure"
    },
    {
      "description": "Resolvent builds on the same L0 operator, input/output maps, and energy-norm weights from assembly.",
      "from_domain": "Resolvent Analysis",
      "relation_type": "Data Dependency",
      "strength": 9.0,
      "to_domain": "LNSE Operator Assembly"
    },
    {
      "description": "Map walks and stable sweeps call resolvent gain extraction while reusing factors across frequencies.",
      "from_domain": "Solver Routing and Sweep Optimization",
      "relation_type": "Service Call",
      "strength": 9.0,
      "to_domain": "Resolvent Analysis"
    },
    {
      "description": "Anchors, preconditioned sweeps, and residual-gated refactor decisions center on factorization cost and reuse.",
      "from_domain": "Solver Routing and Sweep Optimization",
      "relation_type": "Service Call",
      "strength": 9.5,
      "to_domain": "Linear Algebra Infrastructure"
    },
    {
      "description": "Router assembles or reuses one LNSE operator per spanwise wavenumber beta before frequency walking.",
      "from_domain": "Solver Routing and Sweep Optimization",
      "relation_type": "Data Dependency",
      "strength": 8.5,
      "to_domain": "LNSE Operator Assembly"
    },
    {
      "description": "Mesh filters and distributed operator paths use shared SciPy–PETSc conversion utilities.",
      "from_domain": "Mesh Discretization",
      "relation_type": "Tool Support",
      "strength": 8.0,
      "to_domain": "Shared Scientific Utilities"
    },
    {
      "description": "BaseFlow fields are defined on the structured (ny,nx) computational mesh geometry.",
      "from_domain": "Baseflow Management",
      "relation_type": "Data Dependency",
      "strength": 8.0,
      "to_domain": "Mesh Discretization"
    },
    {
      "description": "Validated config objects flow into the driver as the single policy source for all solver subsections.",
      "from_domain": "Configuration Management",
      "relation_type": "Configuration Dependency",
      "strength": 7.0,
      "to_domain": "Case Orchestration"
    },
    {
      "description": "Eigen services use PETSc init helpers, cfg mirrors, and residual/metrics utilities.",
      "from_domain": "Stability Analysis",
      "relation_type": "Tool Support",
      "strength": 7.5,
      "to_domain": "Shared Scientific Utilities"
    },
    {
      "description": "Resolvent paths share PETSc vector helpers and init-order constraints with the rest of the stack.",
      "from_domain": "Resolvent Analysis",
      "relation_type": "Tool Support",
      "strength": 7.5,
      "to_domain": "Shared Scientific Utilities"
    },
    {
      "description": "Factorization backends build on core PETSc matrix/vector conversion and initialization utilities.",
      "from_domain": "Linear Algebra Infrastructure",
      "relation_type": "Tool Support",
      "strength": 8.0,
      "to_domain": "Shared Scientific Utilities"
    }
  ]
}
```

### Workflow Research Report
Contains static analysis results of the codebase and business process analysis.

```json
"# System Workflow Analysis\n\n## 1. Main Workflow\n- **Workflow Name**: Temporal Eigenvalue Stability Analysis\n- **Description**: End-to-end hydrodynamic stability analysis for compressible flows. The workflow loads CFD mean-flow data onto a structured mesh, assembles the linearized Navier–Stokes (LNSE) operator with boundary conditions, factorizes shifted systems, and computes temporal (and optionally adjoint) eigenvalues and eigenvectors with PETSc/SLEPc. Results support transition prediction, receptivity studies, and flow-control design. This is the primary scientific product path of LinStabPy and the reference orchestration pattern for other analysis modes.\n- **Flow Diagram**:\n```mermaid\nflowchart TD\n    Start[Case Launch] --> InitRuntime[Initialize PETSc and MPI]\n    InitRuntime --> ResolveConfig[Merge and Validate YAML Config]\n    ResolveConfig --> LoadBaseflow[Load CFD Mean-Flow Fields]\n    LoadBaseflow --> PreprocessBF[Preprocess Sponge Viscosity Shock Cleanup]\n    PreprocessBF --> BuildMesh[Build Structured Mesh and SBP Operators]\n    BuildMesh --> GenCoeffs[Compute LNSE Physical Coefficients]\n    GenCoeffs --> AssembleL0[Assemble Sparse L0 Operator]\n    AssembleL0 --> ApplyBCs[Apply Dirichlet Neumann BC Recipes]\n    ApplyBCs --> FactorShift[Factorize Shifted LNSE System]\n    FactorShift --> EigenSolve[SLEPc Temporal Adjoint Eigen Solve]\n    EigenSolve --> ValidateModes[Validate Modes Residuals Weights]\n    ValidateModes --> EmitPayload[Emit Spectra Modes Metrics Provenance]\n    EmitPayload --> End[Case Complete]\n```\n- **Key Steps**:\n  1. **Runtime initialization**, Start PETSc/MPI, logging, and distributed case orchestration (`driver/run.py`).\n  2. **Configuration resolution**, Merge YAML, environment, and defaults; validate eigen/mesh/MPI policy (`config/typed.py`, `config/schema.py`).\n  3. **Baseflow ingest**, Load npy/npz primitives (`rho`, `u`, `v`, `w`, `T`, sponge) into `BaseFlow` and apply preprocess/transport fill.\n  4. **Mesh discretization**, Create/deform structured mesh; attach SBP finite-difference operators, filters, and runtime state.\n  5. **Coefficient generation**, Derive LNSE coefficients for Cartesian or cylindrical compressible flow from mean-flow fields.\n  6. **Operator assembly and BCs**, Build the large sparse `L0` system for state `q = [rho, u, v, w, T]`; enforce declarative Dirichlet/Neumann recipes and free-DOF sets.\n  7. **Shift-invert factorization**, Factorize shifted LNSE operators (default MUMPS/LU; optional cuDSS) for factorize-once, solve-many use.\n  8. **Eigenanalysis**, Run SLEPc EPS for temporal and/or adjoint spectra; validate modes and residuals; optionally compute structural sensitivity.\n  9. **Result emission**, Package eigenvalues, eigenvectors, solver metrics, and provenance into the case payload.\n\n---\n\n## 2. Other Important Workflows\n\n### 2.1 Resolvent Gain Analysis\n- **Description**: Constructs the resolvent operator \\(H = C\\,(L_0 - i\\omega I)^{-1}\\,B\\) at target frequencies, extracts optimal forcing/response pairs and energy gains via a weighted normal eigenproblem, and reports residual-validated amplifier/receptivity results. Shares the same baseflow–mesh–`L0` preparation path as temporal stability, then specializes in repeated shifted solves and SLEPc gain extraction.\n- **Flow Diagram**:\n```mermaid\nflowchart TD\n    Start[Resolvent Case Start] --> InitRuntime[Initialize Runtime]\n    InitRuntime --> ResolveCfg[Resolve Frequency Weights Backend Policy]\n    ResolveCfg --> PrepMean[Provide Mesh-Aligned BaseFlow]\n    PrepMean --> SBPOps[Supply SBP Diff and Norm Operators]\n    SBPOps --> AssembleL0[Assemble L0 with BCs and Energy Maps]\n    AssembleL0 --> FactorOmega[Factor L0 minus i omega I]\n    FactorOmega --> BuildH[Build Resolvent H and H Hermitian Actions]\n    BuildH --> WeightedShell[Form Weighted Normal Operator Shell]\n    WeightedShell --> GainEPS[SLEPc Extract Gains and Mode Pairs]\n    GainEPS --> ResidualCheck[Validate Residuals]\n    ResidualCheck --> PackageOut[Package Gains Modes Metrics]\n    PackageOut --> End[Resolvent Complete]\n```\n- **Key Steps**:\n  1. Resolve resolvent frequency, energy weights, and solver backend policy.\n  2. Reuse baseflow, mesh/SBP, and BC-enforced `L0` assembly.\n  3. Factor \\((L_0 - i\\omega I)\\) for repeated resolvent applies.\n  4. Apply forward/adjoint resolvent actions with input/output maps `B` and `C`.\n  5. Solve the weighted eigenproblem on \\(H^H W H\\) (MatShell) with SLEPc.\n  6. Validate residuals and emit gains, forcing/response shapes, and metrics.\n\n### 2.2 Beta–Frequency Resolvent Gain Map\n- **Description**: Large parametric HPC campaign that assembles one LNSE operator per spanwise wavenumber \\(\\beta\\), marks frequency bands, and walks each frequency line with measured anchor factorizations, preconditioner reuse, and residual-gated refactorization. Leading gains are recorded across the \\((\\beta, \\omega)\\) map with cost samples for routing feedback. This is the cost-critical path for production resolvent studies.\n- **Flow Diagram**:\n```mermaid\nflowchart TD\n    Start[Map Campaign Launch] --> ValidateMap[Validate Map Router Resource Policy]\n    ValidateMap --> LoopBeta[For Each Spanwise Beta]\n    LoopBeta --> AssembleBeta[Assemble LNSE Operator for Beta]\n    AssembleBeta --> MarkBands[Mark Frequency Bands]\n    MarkBands --> ScheduleWalk[Schedule Frequency Line Walk]\n    ScheduleWalk --> AnchorFactor[Factor Measured Anchors]\n    AnchorFactor --> WalkFreq[Walk Frequencies with PC Reuse]\n    WalkFreq --> ResidualGate{Residual Gate OK?}\n    ResidualGate -->|Yes| ComputeGain[Compute Leading Resolvent Gain]\n    ResidualGate -->|No| Refactor[Refactor Anchor and Continue]\n    Refactor --> ComputeGain\n    ComputeGain --> MoreFreq{More Frequencies?}\n    MoreFreq -->|Yes| WalkFreq\n    MoreFreq -->|No| MoreBeta{More Betas?}\n    MoreBeta -->|Yes| LoopBeta\n    MoreBeta -->|No| RecordMap[Record Map Cost Samples Certification]\n    RecordMap --> End[Map Complete]\n```\n- **Key Steps**:\n  1. Validate resolvent-map, router, and resource sections of the case config.\n  2. Assemble one BC-enforced LNSE operator per \\(\\beta\\).\n  3. Mark frequency bands and schedule line walks.\n  4. Pay real factorization cost at anchors; reuse factors as right preconditioners nearby.\n  5. Residual-gate forces refactor when accuracy degrades; optional stable-regime sweeps apply.\n  6. Select MUMPS/LU/cuDSS backends under RAM/cost policy.\n  7. Extract leading gain per scheduled frequency; record map results and certification metadata.\n\n### 2.3 Spatial Stability Eigenproblem\n- **Description**: Extends prepared LNSE coefficients into spatial block operators and solves generalized spatial eigenvalue problems with SLEPc. Used for streamwise wavenumber and spatial growth/decay analysis. Configuration falls back from `solver.spatial` to `solver.eigen` when spatial-specific YAML is absent.\n- **Flow Diagram**:\n```mermaid\nflowchart TD\n    Start[Spatial Eigen Start] --> ResolveSpatial[Resolve solver.spatial Policy]\n    ResolveSpatial --> SupplyBF[Supply Mean-Flow Fields]\n    SupplyBF --> MeshIndex[Provide Mesh Indexing and SBP]\n    MeshIndex --> SpatialOps[Assemble Spatial Block Operators]\n    SpatialOps --> FactorBackend[Configure Shift-Invert Factor Backend]\n    FactorBackend --> SpatialEPS[Run Generalized SLEPc Spatial EPS]\n    SpatialEPS --> ReturnModes[Return Spatial Modes and Metadata]\n    ReturnModes --> End[Spatial Eigen Complete]\n```\n- **Key Steps**:\n  1. Resolve spatial solver policy (with eigen fallback).\n  2. Reuse baseflow and mesh/SBP infrastructure.\n  3. Build spatial stability block operators (SciPy and PETSc paths).\n  4. Configure direct shift-invert factorization backend.\n  5. Solve generalized spatial eigenproblem and return modes.\n\n### 2.4 Baseflow Ingest and Operator Prep\n- **Description**: Shared preparation pipeline from on-disk CFD dumps through `BaseFlow` preprocess, mesh/SBP setup, coefficient generation, BC application, and finalized `L0` assembly. Runs as preflight/setup before any eigen or resolvent solve and is the common foundation of all analysis workflows.\n- **Flow Diagram**:\n```mermaid\nflowchart TD\n    Start[Prep Stage Start] --> SchemaCheck[Validate Case Schema and FD Orders]\n    SchemaCheck --> LoadFields[Ingest npy or npz Primitives]\n    LoadFields --> Preprocess[Shock Cleanup Sponge Viscosity Fill]\n    Preprocess --> MeshCreate[Create or Deform Mesh Attach SBP Filters]\n    MeshCreate --> Coeffs[Generate LNSE Coefficients]\n    Coeffs --> BCRecipe[Apply BC Recipes Free-DOF Sets]\n    BCRecipe --> FinalizeL0[Finalize Sparse L0 for Downstream Solves]\n    FinalizeL0 --> End[Operator Ready]\n```\n- **Key Steps**:\n  1. Schema-validate merged case dictionaries and finite-difference orders.\n  2. Load rectilinear field sets and optional sponge data.\n  3. Apply shock cleanup, sponge, and viscosity/transport helpers.\n  4. Build mesh geometry, SBP stencils (orders 2–8), filters, and masking.\n  5. Sequence declarative BC rules onto the operator.\n  6. Finalize SciPy CSC or PETSc `Mat` `L0` for stability or resolvent stages.\n\n### 2.5 Adjoint and Structural Sensitivity (Supporting Path)\n- **Description**: Builds on the temporal eigen path by computing adjoint modes and combining direct/adjoint information into structural sensitivity fields. Supports flow-control and receptivity insight without a separate full case pipeline, invoked as an extension of stability eigenvalue services after direct modes are available.\n- **Flow Diagram**:\n```mermaid\nflowchart TD\n    Start[Direct Modes Available] --> AdjointSolve[Run Adjoint Eigen Solve]\n    AdjointSolve --> CombineModes[Combine Direct and Adjoint Modes]\n    CombineModes --> Sensitivity[Compute Structural Sensitivity]\n    Sensitivity --> EmitSens[Emit Sensitivity Fields and Metadata]\n    EmitSens --> End[Sensitivity Complete]\n```\n\n---\n\n## 3. Workflow Insights\n\n- **Staged scientific pipeline, not a service mesh**, Operational flow is consistently: *config → baseflow → mesh/SBP → coefficients → L0 + BCs → factorize → eigen/resolvent → payload*. Case orchestration (`driver/run.py`) is the single business hub; library import stays lightweight and does not initialize PETSc/MPI until a real solve path runs.\n\n- **One operator core, many analysis products**, Temporal eigen, adjoint/sensitivity, resolvent gains, spatial eigenproblems, and \\(\\beta\\)–\\(\\omega\\) maps all consume the same BC-enforced LNSE operator family. Spatial assembly reuses temporal preparation helpers; resolvent adds frequency shifts, `B`/`C` maps, and energy-norm weights rather than a separate physics stack.\n\n- **Factorize-once, solve-many is the performance spine**, Shift-invert eigen solves and resolvent \\(H\\) actions both depend on the factorization layer (default KSPPREONLY + PCLU + MUMPS; optional iterative policy or cuDSS). Map campaigns amplify this pattern: anchors pay factorization cost; nearby frequencies reuse factors as preconditioners until a residual gate forces refactor.\n\n- **Cost-aware routing is a first-class workflow**, The router layer (band marking, line walks, stable sweeps, break-even logic, RAM autoselect, certification) turns single-point resolvent analysis into scalable parametric studies. Optimization opportunity: feed measured factor time/memory/compression samples back into anchor placement and backend choice more aggressively across multi-node runs.\n\n- **Strong domain boundaries with clear dependencies**,  \n  - *Case Orchestration* depends on Configuration, Baseflow, Operator Assembly, Stability, Resolvent, Router, and Linear Algebra.  \n  - *LNSE Operator Assembly* composes Mesh and Baseflow, then feeds Stability, Resolvent, and Router.  \n  - *Stability* and *Resolvent* both service-call Linear Algebra; Router service-calls both Resolvent and Linear Algebra while data-depending on per-`\\(\\beta\\)` operators.  \n  - Shared utilities (SciPy↔PETSc bridge, provenance, cost metrics) underpin mesh and solver layers without owning business flow.\n\n- **External system contract is narrow and file-driven**, Inputs are YAML case policy plus CFD mean-flow dumps (npy/npz). Outputs are spectra, modes, gains, metrics, and provenance payloads. Nonlinear CFD, GUIs, unstructured 3D frameworks, and cloud multi-tenancy remain out of scope, keeping workflows reproducible on MPI clusters for researchers and aerospace engineers.\n\n- **Configuration is the reproducibility gate**, Typed config merge plus lenient-but-structured schema validation runs before heavy stages. Invalid FD orders or unknown top-level keys fail fast (`ConfigValidationError`), which protects expensive HPC factorizations and map walks.\n\n- **Potential optimization and extension points**,  \n  1. Deeper reuse of a single prepared `L0` across mixed temporal + resolvent stages in one case.  \n  2. Tighter coupling of residual-gate thresholds to backend-specific (MUMPS BLR / cuDSS) accuracy profiles.  \n  3. Parallelism across \\(\\beta\\) lines where memory allows, while keeping per-line anchor walks sequential for factor reuse.  \n  4. Richer structural-sensitivity packaging in the standard payload path for control-oriented users.\n\n- **Terminology alignment**, Implementation matches documented LinStab2D-style business processes: temporal/spatial/adjoint eigenanalysis, resolvent gain maps, SBP structured discretization, and declarative BC recipes. No material gap was identified between the described business flows and the code-centered stage sequence; boundary conditions are packaged separately in code but functionally belong to operator assembly, which the workflows above reflect."
```

### Code Insights Data
Code analysis results from preprocessing phase, including definitions of functions, classes, and modules.

```json
{
  "directory_insights": [
    {
      "file_count": 0,
      "file_insights": [
        {
          "code_purpose": "doc",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "PETSc",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "SLEPc",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "mumps",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "install.sh",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Explains how linstabpy uses PETSc LU factorization for sparse-direct solves and keeps MUMPS as the reference backend for temporal/spatial EVP, resolvent, and linear-solve validation. It also describes install.sh's one-venv-per-stack isolation so non-default PETSc/SLEPc builds do not collide with the CPU MUMPS .venv reference.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/docs/COMPUTE_BACKENDS.md",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "COMPUTE_BACKENDS.md",
          "responsibilities": [
            "Document default and alternate PETSc factor backends",
            "Describe reference paths for EVP and linear solves",
            "Explain venv isolation per PETSc/SLEPc build stack",
            "Guide maintainers on install.sh backend layout"
          ],
          "source_summary": "Markdown sections cover compute backends and a build-stack isolation strategy. Content states default factor backend mumps, lists solver use cases, and notes side-by-side .venv-<stack> directories for non-default stacks versus the default .venv.",
          "summary": "Documents PETSc/SLEPc sparse-direct compute backends, with MUMPS as the default reference path and isolated virtualenvs per build stack."
        },
        {
          "code_purpose": "doc",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "main",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "dev",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Captures how linstabpy is developed in small verified increments toward 1.0.0 with one field loader, one YAML shape, and no compatibility shims. It specifies main as stable and PR-only, dev as the integration base, and topic branches for features, fixes, and refactors.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/docs/DEVELOPMENT.md",
          "importance_score": 0.5,
          "interfaces": [],
          "name": "DEVELOPMENT.md",
          "responsibilities": [
            "Define branch roles and merge rules",
            "Document release and stability expectations",
            "Align contributors on 1.0.0 API goals",
            "Describe incremental verified development practice"
          ],
          "source_summary": "Markdown workflow doc with a branches table for main, dev, and topic branches (feat/*, fix/*, ref*). States no direct pushes to main, full-suite stability on main, and release tagging from main after reviewed merges from dev.",
          "summary": "Defines the branch model and incremental release workflow targeting a clean 1.0.0 public API surface."
        },
        {
          "code_purpose": "doc",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.boundary_conditions.recipe",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.operators.assembly",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Tells maintainers where boundary-condition recipes, LNSE assembly, and related numerical policy must live. Emphasizes keeping policy in owning modules rather than recreating thin pass-through files, which stabilizes long-term structure of the scientific codebase.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/docs/architecture.md",
          "importance_score": 0.65,
          "interfaces": [],
          "name": "architecture.md",
          "responsibilities": [
            "Map numerical domains to owning modules",
            "Prevent duplicated pass-through policy files",
            "Document BC recipe and assembly responsibilities",
            "Orient maintainers to package structure"
          ],
          "source_summary": "Markdown ownership table maps areas such as boundary-condition recipes to linstabpy.boundary_conditions.recipe and LNSE assembly to linstabpy.operators.assembly, listing responsibilities like YAML recipe parsing, Dirichlet row handling, coefficient assembly, and SciPy/PETSc/spatial dispatch.",
          "summary": "Architecture map that assigns numerical policy ownership to specific linstabpy modules to avoid pass-through duplication."
        },
        {
          "code_purpose": "doc",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.driver.io.save_results",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.driver.payload.RUNTIME_NPZ_CONTRACT_KEYS",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "build_runtime_payload",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Describes how the shared driver writes one .npz per run via linstabpy.driver.io.save_results and treats key names as a compatibility surface. Points to RUNTIME_NPZ_CONTRACT_KEYS and build_runtime_payload as the checked grouping layer before io.py persistence, allowing extra arrays while protecting core keys.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/docs/runtime_npz_contract.md",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "runtime_npz_contract.md",
          "responsibilities": [
            "Define stable NPZ key compatibility surface",
            "Document runtime payload build and save path",
            "Support parity, plotting, and campaign consumers",
            "Clarify allowed extra arrays versus contract keys"
          ],
          "source_summary": "Markdown contract note covering save_results, RUNTIME_NPZ_CONTRACT_KEYS in linstabpy.driver.payload, and build_runtime_payload grouping. States archives support parity checks, plotting scripts, and campaign summaries, and that runtime outputs may include extra arrays beyond the contract set.",
          "summary": "Specifies the shared runtime compressed NPZ output contract used as the handoff between computations, parity checks, and plotting."
        }
      ],
      "importance_score": 0.42,
      "key_files": [
        "architecture.md",
        "runtime_npz_contract.md",
        "COMPUTE_BACKENDS.md",
        "DEVELOPMENT.md"
      ],
      "name": "docs",
      "path": "/home/rfrantz/openfluids/linstabpy/docs",
      "purpose": "docs",
      "subdirectory_count": 2,
      "summary": "The docs directory holds maintainer-facing project documentation for linstabpy, covering numerical compute backends, development/release workflow, module ownership architecture, and the runtime NPZ output contract. Together these notes define how solvers are built and selected, how contributions land on main, where numerical policy lives in code, and which keys form the stable handoff between runs and downstream tools. They do not implement business logic but encode critical operational and compatibility policy for the scientific stack."
    },
    {
      "file_count": 0,
      "file_insights": [
        {
          "code_purpose": "doc",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "docs/assets/readme-banner-v2.jpg",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "xAI Grok CLI image_gen",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This markdown note describes readme-banner-v2.jpg, including resolution, aspect ratio, file size, and that it replaces an older PNG banner. It records the charcoal ground, off-white wordmark, and cyan/teal/coral palette shared with sibling openfluids projects. It is reference documentation for branding assets, not executable code.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/docs/assets/readme-banner-v2.md",
          "importance_score": 0.12,
          "interfaces": [],
          "name": "readme-banner-v2.md",
          "responsibilities": [
            "Document banner v2 metadata and dimensions",
            "Record generation tooling and compositing approach",
            "Describe shared cross-repo visual language",
            "Note replacement of the prior banner asset"
          ],
          "source_summary": "Contains a title, asset path and dimensions, tool/model notes (xAI Grok CLI image_gen plus local compositing), and prose on visual consistency with dynachaos, fftkit, chaos-atlas, openmodalpy, and dsgbr. Mentions README relative-path referencing. No functions or imports.",
          "summary": "Documents the v2 README banner asset path, size, generation tooling, and shared visual language across related repositories."
        },
        {
          "code_purpose": "doc",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "readme-banner-v3.jpg",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "xAI Grok CLI image_gen",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file specifies readme-banner-v3.jpg at 1408x469 (3:1) and how it was produced with Grok image generation plus local compositing. It includes a detailed subject prompt for luminous eigenvalue branches and a neutral stability axis. It supports reproducible branding art rather than runtime behavior.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/docs/assets/readme-banner-v3.md",
          "importance_score": 0.14,
          "interfaces": [],
          "name": "readme-banner-v3.md",
          "responsibilities": [
            "Document banner v3 asset identity and size",
            "Capture image generation subject prompt",
            "Record tooling used to produce the banner",
            "Support consistent README visual presentation"
          ],
          "source_summary": "Markdown with asset name/size, tooling notes, and a fenced subject prompt describing a complex-plane eigenvalue spectrum with fine luminous points, sweeping branches, and a horizontal stability axis. Content is truncated in the provided excerpt. No code interfaces.",
          "summary": "Documents the v3 README banner, including dimensions and the subject prompt for an eigenvalue spectrum visualization in the complex plane."
        }
      ],
      "importance_score": 0.15,
      "key_files": [
        "readme-banner-v3.md",
        "readme-banner-v2.md"
      ],
      "name": "assets",
      "path": "/home/rfrantz/openfluids/linstabpy/docs/assets",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The assets directory holds documentation describing README banner image assets used across related openfluids repositories. The markdown files record provenance, dimensions, visual language, and generation prompts for banner versions rather than application logic. Together they serve as lightweight asset notes for branding consistency in project documentation."
    },
    {
      "file_count": 0,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "pdflatex",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "include",
              "is_external": false,
              "line_number": null,
              "name": "tutorial.tex",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "rm",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines the primary build pipeline for the tutorial document using pdflatex with nonstop and halt-on-error flags. The default target produces tutorial.pdf by running the compiler twice, and a clean target removes aux, log, out, listing, and pdf outputs.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/docs/tutorial/Makefile",
          "importance_score": 0.2,
          "interfaces": [
            {
              "description": null,
              "interface_type": "make_target",
              "name": "all",
              "parameters": [],
              "return_type": "void",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "make_target",
              "name": "clean",
              "parameters": [],
              "return_type": "void",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "make_target",
              "name": "$(TARGET).pdf",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "$(TARGET).tex",
                  "param_type": "file"
                }
              ],
              "return_type": "file",
              "visibility": ""
            }
          ],
          "name": "Makefile",
          "responsibilities": [
            "Define LaTeX compiler and flags",
            "Build tutorial.pdf from tutorial.tex",
            "Run pdflatex twice for stable references",
            "Remove intermediate and output build files"
          ],
          "source_summary": "Sets TEX=pdflatex and TEXFLAGS, TARGET=tutorial. The all target depends on tutorial.pdf; tutorial.pdf depends on tutorial.tex and invokes pdflatex twice. The clean rule deletes common LaTeX build byproducts; all and clean are marked .PHONY.",
          "summary": "GNU Make build file that compiles the tutorial LaTeX source into a PDF and cleans generated artifacts."
        }
      ],
      "importance_score": 0.15,
      "key_files": [
        "Makefile"
      ],
      "name": "tutorial",
      "path": "/home/rfrantz/openfluids/linstabpy/docs/tutorial",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The tutorial directory provides build automation for a LaTeX-based tutorial document. Its Makefile compiles tutorial.tex into tutorial.pdf with pdflatex (run twice for references) and exposes clean targets for auxiliary artifacts. It is documentation/build infrastructure rather than application business logic."
    },
    {
      "file_count": 3,
      "file_insights": [
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Serves as the primary import surface for linstabpy, describing stability, adjoint, structural sensitivity, and resolvent workflows. It defers heavy solver initialization so a plain import stays lightweight, loading public symbols only on first attribute access. This makes it the hub consumers and subpackages rely on for version and API discovery.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/__init__.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_package_version",
              "parameters": [],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__getattr__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "object",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__dir__",
              "parameters": [],
              "return_type": "list[str]",
              "visibility": ""
            }
          ],
          "name": "__init__.py",
          "responsibilities": [
            "Define package public API surface",
            "Lazy-load public symbols on first use",
            "Report package version without heavy imports",
            "Avoid eager PETSc/MPI initialization"
          ],
          "source_summary": "Defines package docstring for compressible viscous flow stability analysis and implements lazy attribute resolution. Exposes _package_version, __getattr__, and __dir__ so public names resolve on demand without eager PETSc/MPI startup. SPDX GPL-3.0-only license header is present.",
          "summary": "Package entry that documents the library purpose and exposes public names via lazy loading without starting PETSc or MPI on import."
        },
        {
          "code_purpose": "types",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Centralizes the contract for run telemetry so rank-0 writers emit consistent <stem>.metrics.json next to saved NPZ results. SCHEMA groups keys into provenance, problem, cost, and quality so validation errors can name the failing group. Units for wall time and memory are documented to keep cost and quality comparisons reproducible.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/metrics_schema.py",
          "importance_score": 0.78,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FieldSpec",
              "parameters": [],
              "return_type": "FieldSpec",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "validate_metrics",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "d",
                  "param_type": "Mapping[str, Any]"
                }
              ],
              "return_type": "list[str]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_type_name",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "types",
                  "param_type": "TypeSpec"
                }
              ],
              "return_type": "str",
              "visibility": ""
            }
          ],
          "name": "metrics_schema.py",
          "responsibilities": [
            "Declare metrics field specifications",
            "Validate metrics dictionaries against schema",
            "Group keys into provenance/problem/cost/quality",
            "Report human-readable type names on errors"
          ],
          "source_summary": "Provides FieldSpec, validate_metrics over a Mapping, and _type_name helpers for type-spec reporting. Documents wall-time seconds and memory-byte fields such as cost_wall_*_s and cost_peak_rss_bytes. Focuses on flat-dict validation rather than runtime emission logic.",
          "summary": "Defines and validates the flat JSON metrics schema written beside each run NPZ for provenance, problem, cost, and quality records."
        },
        {
          "code_purpose": "config",
          "dependencies": [],
          "detailed_description": "Holds a small constant table so configuration and PETSc integration agree on MUMPS ordering without importing PETSc at module load. Supports amd, pord, metis, qamd, and auto orderings via MUMPS_ORDERING_TO_ICNTL_7. Acts as a thin shared config primitive rather than business logic.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/mumps_options.py",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "mumps_options.py",
          "responsibilities": [
            "Map MUMPS ordering names to ICNTL codes",
            "Share constants without PETSc dependency",
            "Provide stable config keys for solver setup"
          ],
          "source_summary": "Exports a single dict MUMPS_ORDERING_TO_ICNTL_7 mapping string ordering names to ICNTL(7) values 0, 4, 5, 6, and 7. Declares __all__ for that symbol only. Uses from __future__ import annotations and GPL-3.0-only SPDX header.",
          "summary": "PETSc-free mapping of human-readable MUMPS ordering names to ICNTL(7) integer codes shared by config and solver wiring."
        }
      ],
      "importance_score": 0.92,
      "key_files": [
        "__init__.py",
        "metrics_schema.py",
        "mumps_options.py"
      ],
      "name": "linstabpy",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy",
      "purpose": "other",
      "subdirectory_count": 12,
      "summary": "linstabpy is the root Python package for stability and resolvent analysis of compressible viscous flows, assembling linearized Navier-Stokes operators on structured meshes with SBP finite differences and PETSc/SLEPc/MUMPS backends. The three top-level files define lazy public exports, run metrics schema validation, and shared MUMPS ordering constants used by config and solver wiring. Together they form the package entry surface and cross-cutting configuration/types layer above the numerical submodules."
    },
    {
      "file_count": 7,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "baseflow.BaseFlow",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "grid_fields",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "preprocess.apply_preprocess",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "sponge",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "viscosity.sutherland_air",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines the public surface of the baseflow subpackage under GPL-3.0. It aggregates BaseFlow, field loaders, preprocess hooks, sponge helpers, and the Sutherland viscosity routine so callers can import from a single namespace.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/baseflow/__init__.py",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "__init__.py",
          "responsibilities": [
            "Expose public baseflow API",
            "Aggregate submodule exports",
            "Document package purpose"
          ],
          "source_summary": "A short module docstring and from-import statements expose BaseFlow, load_case_fields, load_grid_fields, read_grid_fields, apply_preprocess, generate_sponge, load_sponge_aux, sponge_from_config, and sutherland_air. No local logic beyond re-export.",
          "summary": "Package entry that re-exports the public base-flow API for consumers of linstabpy.baseflow."
        },
        {
          "code_purpose": "model",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "dataclasses.dataclass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "BaseFlow is the central data model for mean-flow state used throughout the stability pipeline. All primitive and derived arrays are expected on a (ny, nx) computational mesh, and viscosity helpers later fill transport coefficients in place.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/baseflow/baseflow.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "BaseFlow",
              "parameters": [],
              "return_type": "BaseFlow",
              "visibility": ""
            }
          ],
          "name": "baseflow.py",
          "responsibilities": [
            "Represent base-flow fields and parameters",
            "Enforce mesh-shaped array conventions",
            "Serve as shared container for loaders and viscosity models"
          ],
          "source_summary": "Uses a dataclass over numpy arrays to store base-flow fields and parameters. The class documents shape conventions and is designed to be mutated by sutherland_air and loaders rather than recomputed from scratch.",
          "summary": "Defines the BaseFlow dataclass that holds mesh-aligned flow fields and parameters for linear stability analysis."
        },
        {
          "code_purpose": "dao",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib.Path",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "baseflow.BaseFlow",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "preprocess.apply_preprocess",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements I/O for case and grid field sets, validating 2-D float arrays and known field names such as x, y, rho, u, v, w, T, and sponge. It bridges on-disk CFD mean-flow dumps into the in-memory BaseFlow workflow and can apply preprocess steps after load.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/baseflow/grid_fields.py",
          "importance_score": 0.88,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_as_field",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "arr",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "Path"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "read_grid_fields",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str | Path"
                }
              ],
              "return_type": "dict[str, np.ndarray]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "load_grid_fields",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "load_case_fields",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "grid_fields.py",
          "responsibilities": [
            "Read npy/npz field packs",
            "Validate field shapes and dtypes",
            "Map disk fields into analysis structures",
            "Optionally invoke preprocess after load"
          ],
          "source_summary": "Defines FIELD_NAMES and helpers like _as_field and read_grid_fields, plus higher-level load_case_fields/load_grid_fields paths. Arrays are coerced to float, checked for ndim==2, and assembled into dicts consumed by BaseFlow construction and preprocess.",
          "summary": "Loads rectilinear primitive fields from an npy directory or a single npz archive into BaseFlow-ready dictionaries."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "shock_cleanup.apply_optional_shock_cleanup",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides the main apply_preprocess pathway that can smooth fields, build or attach sponge regions, and optionally run shock-aware cleanup. It is the orchestration layer between raw loaded fields and analysis-ready BaseFlow content.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/baseflow/preprocess.py",
          "importance_score": 0.86,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_binomial5_filter",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "arr",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "axis",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_as_four_numbers",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "values",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cast",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "tuple[Any, Any, Any, Any]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_ramp",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "count",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "apply_preprocess",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "preprocess.py",
          "responsibilities": [
            "Apply optional spatial filters",
            "Configure and apply boundary sponges",
            "Normalize preprocess configuration",
            "Delegate optional shock cleanup"
          ],
          "source_summary": "Includes a 5-point binomial filter, ramp helpers, config normalization utilities such as _as_four_numbers, and integration with shock_cleanup.apply_optional_shock_cleanup. Logic is config-driven and operates on 2-D numpy field arrays.",
          "summary": "Optional mean-flow filtering, boundary sponge application, and hooks into shock cleanup before stability analysis."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.ndimage.binary_dilation",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.ndimage.gaussian_filter",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Self-contained experimental cleanup path so Case 1/Case 2 defaults stay untouched unless enabled in config. It normalizes cleanup settings, identifies shock-affected regions via gradients and morphology, and filters selected primitive fields.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/baseflow/shock_cleanup.py",
          "importance_score": 0.78,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_deep_merge",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "base",
                  "param_type": "dict[str, Any]"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "override",
                  "param_type": "dict[str, Any]"
                }
              ],
              "return_type": "dict[str, Any]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_normalize_cfg",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "dict[str, Any] | None"
                }
              ],
              "return_type": "dict[str, Any]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_normalize_directions",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "values",
                  "param_type": "Any"
                }
              ],
              "return_type": "list[int]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_normalize_fields",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "values",
                  "param_type": "Any"
                }
              ],
              "return_type": "list[str]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_binomial5_filter",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "arr",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "axis",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_computational_gradients",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "field",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "tuple[np.ndarray, np.ndarray]",
              "visibility": ""
            }
          ],
          "name": "shock_cleanup.py",
          "responsibilities": [
            "Normalize shock-cleanup configuration",
            "Detect shock-influenced regions",
            "Filter selected primitive fields",
            "Keep cleanup fully opt-in"
          ],
          "source_summary": "Provides config merge/normalization, direction and field validators, binomial filtering, computational gradients, and scipy.ndimage dilation/gaussian smoothing. Valid fields are rho, u, v, w, T; behavior is gated by an enabled flag.",
          "summary": "Opt-in shock-aware mean-flow cleanup that can smooth or repair fields near detected discontinuities."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements sponge construction for stability domains, including localized presets and per-edge counts, amplitudes, widths, and fractions. Supports loading auxiliary sponge data and building sponges from configuration for top/right/bottom/left boundaries.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/baseflow/sponge.py",
          "importance_score": 0.84,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_as_dir_counts",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "values",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_as_dir_amplitudes",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "values",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_as_dir_widths",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "values",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_as_dir_fractions",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "values",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_cumulative_distances",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "X",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "Y",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "axis",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_boundary_distance_fields",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "X",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "Y",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "dict[str, np.ndarray]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_boundary_reference_lengths",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "X",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "Y",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "load_sponge_aux",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "case_inputs_mat",
                  "param_type": "Any"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "generate_sponge",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "sponge_from_config",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "sponge.py",
          "responsibilities": [
            "Generate boundary sponge fields",
            "Parse directional sponge parameters",
            "Provide named sponge presets",
            "Load auxiliary sponge inputs from case data"
          ],
          "source_summary": "Defines EDGE_NAMES, SPONGE_PRESETS, distance-field helpers on X/Y meshes, and public APIs such as generate_sponge, load_sponge_aux, and sponge_from_config. Direction ordering is fixed as [top, right, bottom, left] with axis conventions documented in-module.",
          "summary": "Generates structured sponge/absorbing layers on tensor meshes from presets and directional parameters."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "baseflow.BaseFlow",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Ports sutherland_air.m from the MATLAB LinStab2D codebase into Python. It mutates BaseFlow in place so downstream linear operators can use consistent transport properties and temperature derivatives.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/baseflow/viscosity.py",
          "importance_score": 0.82,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "sutherland_air",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "bf",
                  "param_type": "BaseFlow"
                }
              ],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "viscosity.py",
          "responsibilities": [
            "Compute Sutherland viscosity for air",
            "Derive viscosity temperature derivatives",
            "Set heat-capacity related coefficients on BaseFlow"
          ],
          "source_summary": "Exposes sutherland_air(bf) which fills MU, dmudT, d2mudT2, cv, c1, and c2 from temperature and fluid assumptions for air. Depends only on numpy and the BaseFlow model.",
          "summary": "Applies Sutherland's law for air to populate viscosity and related thermodynamic coefficients on a BaseFlow."
        }
      ],
      "importance_score": 0.9,
      "key_files": [
        "baseflow.py",
        "grid_fields.py",
        "preprocess.py",
        "sponge.py",
        "viscosity.py"
      ],
      "name": "baseflow",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/baseflow",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The baseflow package is a core domain module of linstabpy that defines, loads, and prepares mean-flow fields for linear stability analysis. It centers on the BaseFlow data container and supporting loaders, optional preprocess/shock cleanup, sponge-layer generation, and Sutherland viscosity models that together turn raw grid fields into analysis-ready base-flow state."
    },
    {
      "file_count": 4,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.boundary_conditions.dirichlet",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.boundary_conditions.neumann",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.boundary_conditions.recipe",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines PEP 562 lazy attribute resolution for apply_dirichlet, Neumann helpers, and recipe analysis entry points. This keeps consumers of pure scipy paths from paying petsc4py import cost at package load. It is the public surface of the boundary_conditions subpackage.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/boundary_conditions/__init__.py",
          "importance_score": 0.72,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "__getattr__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__dir__",
              "parameters": [],
              "return_type": "list",
              "visibility": ""
            }
          ],
          "name": "__init__.py",
          "responsibilities": [
            "Lazy public API re-export",
            "Defer PETSc-dependent imports",
            "Expose package __dir__ for discoverability"
          ],
          "source_summary": "Documents lazy export intent and maps public names to submodule paths via _LAZY_EXPORTS. Implements __getattr__ and __dir__ to resolve and list those exports on demand without eager submodule imports.",
          "summary": "Package entry that lazily re-exports public BC APIs so PETSc-backed variants are not imported unless requested."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "petsc4py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Translates MATLAB BC_Dirichlet.m into Python sparse-matrix operations that zero constrained rows/columns and pin Dirichlet DOFs. Provides helpers for spatial index doubling, CSR/CSC mutation, and free-DOF masks used by the rest of the BC stack. Central low-level enforcement path for essential boundary conditions.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/boundary_conditions/dirichlet.py",
          "importance_score": 0.9,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_zero_rows_csr",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mat",
                  "param_type": "sp.csr_matrix"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rows",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_zero_cols_csc",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mat",
                  "param_type": "sp.csc_matrix"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cols",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_spatial_doubled_indices",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "base_indices",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ngp",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_zero_rows_cols",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mat",
                  "param_type": "sp.csc_matrix"
                }
              ],
              "return_type": "sp.csc_matrix",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "dirichlet_free_dofs",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n_dofs",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dirichlet_indices",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            }
          ],
          "name": "dirichlet.py",
          "responsibilities": [
            "Zero Dirichlet rows and columns in sparse operators",
            "Map border/variable symbols to DOF indices",
            "Compute free DOF sets after Dirichlet constraints",
            "Support spatial doubled indexing for multi-field systems"
          ],
          "source_summary": "Contains utilities such as _zero_rows_csr, _zero_cols_csc, _spatial_doubled_indices, _zero_rows_cols, and dirichlet_free_dofs, plus higher-level apply_dirichlet-style APIs. Uses numpy/scipy.sparse and TYPE_CHECKING imports for optional petsc4py support while preserving sparse structure.",
          "summary": "Implements Dirichlet boundary enforcement on scipy sparse LNSE operators, including free-DOF selection and optional PETSc variants."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.mesh.mesh.Mesh",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.boundary_conditions.dirichlet",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Port of MATLAB BC_Neumann.m that replaces boundary operator rows with normal-derivative stencils rather than essential value constraints. Reuses Dirichlet border/variable maps and zeroing helpers so Neumann and Dirichlet share index conventions. Complements Dirichlet enforcement for mixed natural/essential BC setups.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/boundary_conditions/neumann.py",
          "importance_score": 0.86,
          "interfaces": [],
          "name": "neumann.py",
          "responsibilities": [
            "Enforce Neumann conditions on boundary DOFs",
            "Preserve sparse LNSE operator structure",
            "Reuse shared border and variable index maps",
            "Integrate mesh geometry for normal derivatives"
          ],
          "source_summary": "Imports Mesh and shared maps/helpers from dirichlet, then builds Neumann row modifications on scipy sparse matrices. Focuses on normal-derivative operator assembly on boundary DOFs without densifying the global system.",
          "summary": "Applies Neumann (derivative-normal) boundary rows to LNSE operator DOFs in curvilinear coordinates while keeping sparsity."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.mesh.mesh.Mesh",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.boundary_conditions.dirichlet",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "collections.abc",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "dataclasses",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines BCRule and BCRecipe abstractions that validate symbols, analyze mesh index sets, and apply Dirichlet/Neumann rules in order. Acts as the orchestration layer between configuration and low-level matrix enforcement. Highest-level business API in this directory for composing multi-border multi-variable BC policies.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/boundary_conditions/recipe.py",
          "importance_score": 0.92,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "BCRule",
              "parameters": [],
              "return_type": "BCRule",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "from_mapping",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rule",
                  "param_type": "Mapping[str, Any]"
                }
              ],
              "return_type": "BCRule",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "BCRecipe",
              "parameters": [],
              "return_type": "BCRecipe",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "analyze",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mesh",
                  "param_type": "Mesh"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "idx",
                  "param_type": "dict[str, np.ndarray]"
                }
              ],
              "return_type": "dict[str, object]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_validate_rule_symbols",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "bc_type",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "borders",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "variables",
                  "param_type": "str"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_require_spatial_dirichlet_recipe",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "analysis",
                  "param_type": "dict[str, object]"
                }
              ],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "recipe.py",
          "responsibilities": [
            "Parse and validate declarative BC rules",
            "Analyze mesh indices for borders and variables",
            "Apply ordered Dirichlet/Neumann recipes to operators",
            "Return combined BC index and analysis results"
          ],
          "source_summary": "Implements dataclasses/classes BCRule and BCRecipe with from_mapping and analyze methods, plus validators like _validate_rule_symbols and _require_spatial_dirichlet_recipe. Reads rule mappings, drives sequential application via dirichlet/neumann helpers, and returns combined BC analysis dictionaries.",
          "summary": "Declarative BC recipe model and applicator that sequences YAML-style rules onto the LNSE operator and returns combined index sets."
        }
      ],
      "importance_score": 0.88,
      "key_files": [
        "recipe.py",
        "dirichlet.py",
        "neumann.py",
        "__init__.py"
      ],
      "name": "boundary_conditions",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/boundary_conditions",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The boundary_conditions package enforces Dirichlet and Neumann conditions on the LNSE sparse operator and exposes a declarative recipe layer for applying multi-rule BC configs. Lazy package exports keep pure-math paths free of PETSc while dirichlet/neumann implement matrix row/column surgery and recipe orchestrates sequential application from YAML-style rules. Together these modules form core numerical domain logic for LinStab2D-style stability analysis."
    },
    {
      "file_count": 5,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.config.defaults",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.config.schema",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.config.typed",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Acts as the public facade of linstabpy.config so callers import Config, validate_config, and related types from one place. It wires defaults, schema validation, and typed config symbols without implementing logic itself.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/config/__init__.py",
          "importance_score": 0.75,
          "interfaces": [],
          "name": "__init__.py",
          "responsibilities": [
            "Expose package public API",
            "Re-export validation and typed config symbols",
            "Document package purpose via module docstring"
          ],
          "source_summary": "GPL-licensed module docstring describes configuration validation helpers. It imports DISSIPATION_APPLY_DEFAULTS from defaults, ConfigValidationError and validate_config from schema, and Config plus solver/mesh/MPI/resolvent dataclasses from typed, then lists them in __all__.",
          "summary": "Package entry that re-exports defaults, validation API, and typed config dataclasses for consumers."
        },
        {
          "code_purpose": "config",
          "dependencies": [],
          "detailed_description": "Prevents silent operator mismatches between stability and resolvent stages by declaring shared defaults once. Documents a historical bug where dissipation was applied inconsistently across driver stages.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/config/defaults.py",
          "importance_score": 0.82,
          "interfaces": [],
          "name": "defaults.py",
          "responsibilities": [
            "Declare shared config defaults once",
            "Align dissipation apply flags across stages",
            "Document historical mismatch rationale"
          ],
          "source_summary": "Module explains why defaults must be centralized after driver/run.py disagreed on operator_dissipation.apply_to_resolvent versus apply_to_stability. Defines DISSIPATION_APPLY_DEFAULTS and related agreed default values used by multiple layers.",
          "summary": "Single source of truth for cross-layer configuration defaults, especially dissipation apply flags."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "collections.abc.Iterable",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing.Any",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Validates merged case configs before solvers run, rejecting unknown structure while remaining lenient enough for legacy flat keys. Raises ConfigValidationError on invalid configurations and walks nested finite-difference order values.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/config/schema.py",
          "importance_score": 0.9,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "ConfigValidationError",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "validate_config",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "dict"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str | None"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_iter_fd_order_values",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "str"
                }
              ],
              "return_type": "Iterable[tuple[str, Any]]",
              "visibility": ""
            }
          ],
          "name": "schema.py",
          "responsibilities": [
            "Validate merged YAML case dictionaries",
            "Enforce allowed top-level config keys",
            "Report validation errors with optional source context",
            "Iterate and check finite-difference order values"
          ],
          "source_summary": "Defines ConfigValidationError, TOP_LEVEL_ALLOWED frozenset for paths/physics/numerics and related blocks, validate_config(cfg, ..., source), and helper _iter_fd_order_values to traverse FD order settings. Focuses on structural checks for Martini/Song-style case dicts.",
          "summary": "Lenient schema validation for fully merged YAML case dictionaries with allowed top-level keys and FD-order checks."
        },
        {
          "code_purpose": "types",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "collections.abc.Mapping",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "dataclasses.dataclass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib.Path",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing.Any",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.linalg.backends",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Core typed configuration model for linear-solve, resolvent, eigen, mesh, MPI, and related solver subsections. Mirrors utils._solver_cfg resolution so call sites can migrate gradually to typed access without changing behavior.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/config/typed.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_solver_mapping",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "raw",
                  "param_type": "Mapping[str, Any] | None"
                }
              ],
              "return_type": "Mapping[str, Any]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_subsection",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "solver",
                  "param_type": "Mapping[str, Any]"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "prefix",
                  "param_type": "str"
                }
              ],
              "return_type": "Mapping[str, Any]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_factor_precision",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "Mapping[str, Any] | None"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "env",
                  "param_type": "Mapping[str, str]"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "LinSolveCfg",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "ResolventCfg",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "InexactCfg.from_mapping",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "Mapping[str, Any] | None"
                }
              ],
              "return_type": "InexactCfg",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "BackendProbeCfg.from_mapping",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "Mapping[str, Any] | None"
                }
              ],
              "return_type": "BackendProbeCfg",
              "visibility": ""
            }
          ],
          "name": "typed.py",
          "responsibilities": [
            "Provide typed config dataclasses for solver policy",
            "Resolve YAML/env/default precedence for backends and MUMPS options",
            "Parse subsections for resolvent, spatial, eigen, mesh, and MPI",
            "Bridge legacy solver_cfg mapping into structured types"
          ],
          "source_summary": "Imports backend defaults from linstabpy.linalg.backends and defines helpers (_solver_mapping, _subsection, env/precision/MUMPS parsers) plus dataclasses such as LinSolveCfg, ResolventCfg, ResolventEpsCfg, InexactCfg, BackendProbeCfg, MeshCfg, MpiCfg, EigenCfg, SpatialCfg, and Config with from_mapping factories.",
          "summary": "Typed dataclass seam resolving YAML, environment, and default solver policy into structured config objects."
        },
        {
          "code_purpose": "config",
          "dependencies": [],
          "detailed_description": "Lists named validation cases (curvilinear, Blasius resolvent variants, etc.) with module paths and expected artifact outputs. Explicitly not a solver case and does not pass through driver.run.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/config/validation_cases.yaml",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "validation_cases.yaml",
          "responsibilities": [
            "Register packaged validation cases",
            "Map case names to example modules",
            "List expected artifact output paths"
          ],
          "source_summary": "YAML structure under cases: entries with name, module (examples.martini...), and artifacts lists pointing to .npz files. Serves as a catalog for validation packaging rather than runtime solver input.",
          "summary": "Registry of packaged validation example modules and the NPZ artifacts each writes."
        }
      ],
      "importance_score": 0.88,
      "key_files": [
        "typed.py",
        "schema.py",
        "defaults.py",
        "__init__.py",
        "validation_cases.yaml"
      ],
      "name": "config",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/config",
      "purpose": "config",
      "subdirectory_count": 0,
      "summary": "The config package is the typed configuration and validation layer for linstabpy solver cases. It centralizes shared defaults, lenient YAML schema checks, and dataclass-based typed views (mesh, MPI, eigen, resolvent, linear solve) so drivers and numerics agree on operator and solver policy. Public symbols are re-exported from __init__.py; validation_cases.yaml registers packaged example artifacts outside the main run path."
    },
    {
      "file_count": 0,
      "file_insights": [
        {
          "code_purpose": "model",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "MATLAB",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Binary MATLAB Level-5 MAT container holding arrays and variables that represent a Mach-0.9 jet baseflow field. It serves as reference or initial-condition data for downstream analysis or stability computations rather than implementing business logic.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/data/baseflows/M09jet.mat",
          "importance_score": 0.4,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "load",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "filename",
                  "param_type": "string"
                }
              ],
              "return_type": "struct",
              "visibility": ""
            }
          ],
          "name": "M09jet.mat",
          "responsibilities": [
            "Persist baseline jet-flow field variables",
            "Provide loadable numerical inputs for MATLAB workflows",
            "Act as reference dataset for simulation or analysis pipelines"
          ],
          "source_summary": "File header identifies MATLAB 5.0 MAT-file format for GLNXA64, dated Fri Feb 17 15:02:45 2023. Remaining content is compressed or binary numeric payload not human-readable as source code; no functions or scripts are defined inside the file.",
          "summary": "MATLAB 5.0 MAT-file storing baseline jet-flow numerical data created on a Linux x86_64 platform."
        },
        {
          "code_purpose": "model",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "MATLAB MAT-file format",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides serialized baseflow field data associated with a paraboloid geometry, created as a MATLAB 5.0 MAT-file on MACI64. It serves as an input dataset rather than source code, supporting downstream solvers or post-processing that need predefined baseflow states.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/data/baseflows/Paraboloid_BF.mat",
          "importance_score": 0.35,
          "interfaces": [],
          "name": "Paraboloid_BF.mat",
          "responsibilities": [
            "Persist paraboloid baseflow numerical fields",
            "Supply loadable input data for MATLAB or compatible solvers",
            "Encode geometry- or flow-specific reference state"
          ],
          "source_summary": "The file is a MATLAB Level-5 MAT container (Platform MACI64, dated Tue Feb 28 15:21:58 2023) with compressed binary payloads. Content is not human-readable source; it holds numeric arrays and variables typical of baseflow definitions (e.g., velocity or geometric field data) for a paraboloid case.",
          "summary": "Binary MATLAB MAT-file storing paraboloid baseflow numerical data for simulation or analysis workflows."
        }
      ],
      "importance_score": 0.35,
      "key_files": [
        "M09jet.mat",
        "Paraboloid_BF.mat"
      ],
      "name": "baseflows",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/data/baseflows",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains multiple file groups. Key aspects: The baseflows directory holds MATLAB binary MAT-file datasets used as baseline flow fields, likely for fluid-dynamics or jet-flow analysis. In this batch the primary artifact is M09jet.mat, a serialized numerical dataset rather than executable application code. Files act as static scientific inputs consumed by external MATLAB or simulation tooling rather than forming an API or service layer. The baseflows directory holds MATLAB MAT data assets used as base-flow inputs for scientific or engineering simulations. This batch contains a single binary MAT file encoding paraboloid-related baseflow fields rather than executable application logic. Files act as static numerical datasets consumed by analysis or solver code elsewhere in the project."
    },
    {
      "file_count": 6,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [],
          "detailed_description": "Defines the driver package surface with an empty __all__ and explicit guidance to import run only on the execution path. It keeps optional heavy dependencies out of package import time, improving CLI responsiveness and optional-stack isolation.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/driver/__init__.py",
          "importance_score": 0.35,
          "interfaces": [],
          "name": "__init__.py",
          "responsibilities": [
            "Declare package boundary and import policy",
            "Keep PETSc/MPI out of package init",
            "Document explicit run-path import"
          ],
          "source_summary": "Contains SPDX header, module docstring explaining lazy runtime import of linstabpy.driver.run, and an empty __all__ list. No functions or classes are defined.",
          "summary": "Package initializer that documents lazy import boundaries so CLI help/version work without loading PETSc/MPI."
        },
        {
          "code_purpose": "command",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "importlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "importlib.metadata",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib.Path",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils.yaml_io.load_config",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements the user-facing CLI for shared case drivers, including version reporting with source-tree fallback and argparse construction. It loads configuration via yaml_io and is designed so help/version paths avoid eagerly importing the PETSc-backed runtime.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/driver/cli.py",
          "importance_score": 0.82,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_package_version",
              "parameters": [],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_build_parser",
              "parameters": [],
              "return_type": "argparse.ArgumentParser",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "argv",
                  "param_type": "Sequence[str] | None"
                }
              ],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "cli.py",
          "responsibilities": [
            "Build and parse CLI arguments",
            "Report package version with fallback",
            "Load case configuration and invoke runtime"
          ],
          "source_summary": "Defines _package_version using importlib.metadata, _build_parser for argparse options, and main(argv) as the CLI orchestrator. Imports include argparse, numpy, pathlib, and linstabpy.utils.yaml_io.load_config.",
          "summary": "Command-line entry points that parse arguments, resolve package version, load YAML config, and dispatch into the runtime driver."
        },
        {
          "code_purpose": "dao",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "json",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "logging",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib.Path",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing.Any",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides runtime result archive I/O utilities used after solver stages complete. It converts NumPy metrics into JSON-serializable sidecar values and gates root-rank behavior so distributed runs write results safely.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/driver/io.py",
          "importance_score": 0.72,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_jsonify_metrics_value",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_is_root_rank",
              "parameters": [],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "load_results",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "str | Path"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "io.py",
          "responsibilities": [
            "Serialize metrics to JSON-safe forms",
            "Coordinate root-rank result writes",
            "Load persisted runtime results"
          ],
          "source_summary": "Implements _jsonify_metrics_value for NaN/NumPy coercion, _is_root_rank for MPI-aware writes, and load_results for archive reload. Uses json, logging, pathlib, numpy, and typing helpers.",
          "summary": "Helpers for reading and writing runtime result archives, including JSON-safe metric conversion and rank-aware I/O."
        },
        {
          "code_purpose": "model",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing.Any",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Centralizes the NPZ payload schema for stability and resolvent runtime outputs so producers and consumers share one contract. It groups modes, gains, responses, mesh/baseflow fields, and request metadata into a stable archive layout.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/driver/payload.py",
          "importance_score": 0.68,
          "interfaces": [],
          "name": "payload.py",
          "responsibilities": [
            "Define NPZ key contract",
            "Group runtime outputs into archive fields",
            "Stabilize producer/consumer payload shape"
          ],
          "source_summary": "Declares RUNTIME_NPZ_CONTRACT_KEYS covering omega, modes, gains, responses, forcings, resolvent densities, targets, mesh arrays, and related metadata. Focuses on contract assembly rather than heavy control flow.",
          "summary": "Defines the runtime NPZ contract key groups and assembles solver outputs into a structured archive payload."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements non-blocking advisories for configurations that complete successfully yet can yield silently wrong or jointly meaningless results. It complements schema validation with semantic checks, solver-error translation, metric cross-coupling diagnostics, and human-readable banners.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/driver/preflight.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "Advisory",
              "parameters": [],
              "return_type": "Advisory",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "render",
              "parameters": [],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "Mapping[str, Any] | None"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "default",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "config_advisories",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "config",
                  "param_type": "Mapping[str, Any] | None"
                }
              ],
              "return_type": "list[Advisory]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "metric_cross_coupling",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "X",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "Y",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "float",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "translate_solver_error",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "context",
                  "param_type": "Mapping[str, Any] | str | None"
                }
              ],
              "return_type": "Advisory | None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "format_advisories",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "advisories",
                  "param_type": "Sequence[Advisory]"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "title",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "format_banner",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "fields",
                  "param_type": "Mapping[str, Any]"
                }
              ],
              "return_type": "str",
              "visibility": ""
            }
          ],
          "name": "preflight.py",
          "responsibilities": [
            "Detect semantic config traps",
            "Translate solver failures into advisories",
            "Format advisory banners for operators",
            "Compute mesh/metric cross-coupling signals"
          ],
          "source_summary": "Defines Advisory with render methods, config_advisories over nested config paths, metric_cross_coupling on mesh arrays, translate_solver_error, and format helpers. Emphasizes advisory-only design rather than hard validation failures.",
          "summary": "Semantic preflight and post-run advisory system that flags legal-but-misleading configurations without blocking execution."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "threading",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "time",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "contextlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "dataclasses",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib.Path",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "mpi4py.MPI",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "petsc4py.PETSc",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils.petsc_init.init_petsc_from_argv",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.baseflow",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Owns the heavy runtime path: PETSc initialization, MPI coordination, staged logging/heartbeats, config resolution, distributed vector construction, and end-to-end case orchestration. It is the primary business-logic hub connecting baseflow loading, solver stages, payload assembly, and result emission.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/driver/run.py",
          "importance_score": 0.98,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_readonly_empty",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dtype",
                  "param_type": "Any"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_config_hash",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "config_dict",
                  "param_type": "dict"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_first_env",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "names",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "default",
                  "param_type": "str | None"
                }
              ],
              "return_type": "str | None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_runtime_config_path",
              "parameters": [],
              "return_type": "Path",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_output_npz_path",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "provenance",
                  "param_type": "dict[str, str] | None"
                }
              ],
              "return_type": "Path",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_log_prefix",
              "parameters": [],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_log",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "message",
                  "param_type": "str"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_stage_start",
              "parameters": [],
              "return_type": "tuple[float, int]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_log_stage",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "stage",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "stage_start",
                  "param_type": "float | tuple[float, int]"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_default_stage_heartbeat_seconds",
              "parameters": [],
              "return_type": "float",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_parse_alpha_filter",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "alpha_filter_raw",
                  "param_type": "object"
                }
              ],
              "return_type": "float | str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_stage_heartbeat",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "stage",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "interval_seconds",
                  "param_type": "float | None"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_build_distributed_vec_from_global_array",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mat",
                  "param_type": "PETSc.Mat"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "values",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "PETSc.Vec",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_resolve_omega_target",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "raw",
                  "param_type": "object"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "default",
                  "param_type": "complex"
                }
              ],
              "return_type": "complex",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_resolve_optional_float",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "raw",
                  "param_type": "object"
                }
              ],
              "return_type": "float",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_cfg_first",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg_fn",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "paths",
                  "param_type": "tuple"
                }
              ],
              "return_type": "object",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_cfg_get",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "config",
                  "param_type": "dict"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "default",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "_RuntimeConfigSource",
              "parameters": [],
              "return_type": "_RuntimeConfigSource",
              "visibility": ""
            }
          ],
          "name": "run.py",
          "responsibilities": [
            "Initialize PETSc/MPI runtime",
            "Orchestrate multi-stage case pipeline",
            "Resolve runtime config and provenance paths",
            "Provide distributed logging and heartbeats",
            "Build PETSc vectors and drive solver stages"
          ],
          "source_summary": "Large module with many helpers and classes for config hashing, env/path resolution, stage timing, alpha-filter parsing, worker heartbeats, PETSc Vec building, and omega/target resolution. Imports numpy, scipy.sparse, mpi4py, petsc4py, and internal baseflow/runtime utilities after init_petsc_from_argv.",
          "summary": "Core shared case-pipeline orchestration driver that initializes PETSc/MPI and executes the distributed stability/resolvent workflow."
        }
      ],
      "importance_score": 0.93,
      "key_files": [
        "run.py",
        "cli.py",
        "preflight.py",
        "io.py",
        "payload.py"
      ],
      "name": "driver",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/driver",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The driver package is the case-style runtime orchestration layer for linstabpy: it wires CLI entry, PETSc/MPI-backed pipeline execution, result I/O, NPZ payload contracts, and semantic preflight advisories. Lightweight modules (__init__, cli) stay free of heavy solver imports, while run.py owns the distributed stability/resolvent workflow and io/payload/preflight support persistence and user-facing diagnostics. Together these files form the primary execution hub between user configs and the scientific solver stack."
    },
    {
      "file_count": 5,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "backends",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.linalg.factorize",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.linalg.shifted",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.linalg.cudss_factor",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines the public linalg package surface with immediate imports for Backend/available/ensure/select and lazy exports for heavier factorization symbols. It keeps optional GPU and PETSc-heavy modules from loading until first use while documenting the package as linear-algebra seams for reusable factorizations.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/linalg/__init__.py",
          "importance_score": 0.72,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "__getattr__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__dir__",
              "parameters": [],
              "return_type": "list[str]",
              "visibility": ""
            }
          ],
          "name": "__init__.py",
          "responsibilities": [
            "Expose the public linalg API",
            "Lazy-load heavy factorization modules",
            "Re-export backend selection helpers",
            "Maintain package __all__ and attribute discovery"
          ],
          "source_summary": "Uses a _LAZY_EXPORTS map and __getattr__/__dir__ to expose Factorization, factorize, shifted_operator, CudssSolver, and cudss_available. Eagerly imports Backend registry helpers from backends and builds __all__ for the stable public API.",
          "summary": "Package entry that re-exports backend selection APIs and lazily loads factorization, shifted-operator, and cuDSS symbols."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "enum.StrEnum",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "functools.cache",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.mumps_options",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils._solver_cfg",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "PETSc",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Central typed registry for choosing and validating factorization backends used by PETSc-based LU solves. It normalizes backend names, precision, MUMPS ordering/BLR settings, probes availability, and ensures requested backends are present before factorization.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/linalg/backends.py",
          "importance_score": 0.9,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "Backend",
              "parameters": [],
              "return_type": "Backend",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "ensure",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "backend",
                  "param_type": "str | Backend | None"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "available",
              "parameters": [],
              "return_type": "dict[Backend, bool]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "factor_backend_available",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "backend",
                  "param_type": "str"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "ensure_factor_backend_available",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "backend",
                  "param_type": "str"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "backend_probe_is_preconditioner_role",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "st_solver",
                  "param_type": "str | None"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_normalize_backend",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "object"
                }
              ],
              "return_type": "str | None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_petsc",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "backends.py",
          "responsibilities": [
            "Register and normalize factor backends",
            "Probe backend and preconditioner availability",
            "Validate and ensure backend readiness",
            "Normalize MUMPS ordering, BLR, and precision options",
            "Support PETSc LU path configuration"
          ],
          "source_summary": "Defines a Backend StrEnum with _missing_ handling, ensure/available/select-style helpers, and many normalizers for backend strings, factor precision, and MUMPS options. Includes PETSc probe utilities, deterministic RHS filling, safe method calls, and error-code extraction for robust backend selection.",
          "summary": "Sparse-direct factor backend registry and availability probing for PETSc LU paths including MUMPS and related options."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "nvmath",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Wraps nvmath-python cuDSS so resolvent and shift-invert workflows can use GPU sparse-direct solves without PETSc coupling in this module. It handles CSR preparation, optional equilibration, availability checks, and a CudssSolver lifecycle for analysis, numeric factor, refactor on fixed pattern, and RHS solves.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/linalg/cudss_factor.py",
          "importance_score": 0.84,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "cudss_available",
              "parameters": [],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "CudssSolver",
              "parameters": [],
              "return_type": "CudssSolver",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "solve",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rhs",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "refactorize",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kmat",
                  "param_type": "sp.spmatrix"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "free",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "equilibrate",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kmat",
                  "param_type": "sp.csr_matrix"
                }
              ],
              "return_type": "tuple[sp.csr_matrix, np.ndarray, np.ndarray]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "find_threading_layer",
              "parameters": [],
              "return_type": "str | None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_import_nvmath",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "cudss_factor.py",
          "responsibilities": [
            "Detect cuDSS/nvmath availability",
            "Plan and numerically factor sparse CSR matrices on GPU",
            "Refactorize on unchanged sparsity patterns",
            "Solve single right-hand sides",
            "Equilibrate and prepare CSR int32 data"
          ],
          "source_summary": "Implements cudss_available, matrix equilibration, int32 CSR conversion, threading-layer discovery, and CudssSolver with solve/refactorize/free. Helper methods prepare factor data, run tiny probe solves, and read LU nonzero counts from solver info objects.",
          "summary": "PETSc-free NVIDIA cuDSS wrapper providing plan, factorize, refactorize, and solve for GPU sparse-direct factorizations."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "logging",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "PETSc",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.linalg.backends",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.linalg.cudss_factor",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements the primary direct-by-default factorization path (KSPPREONLY + PCLU + MUMPS) plus explicit iterative policy options for memory-bound cases. Factorization encapsulates solve, transpose solve, block solve, refactorize, and destroy, and can shell cuDSS as a PETSc PC when GPU factors are selected.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/linalg/factorize.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "Factorization",
              "parameters": [],
              "return_type": "Factorization",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "solve_transpose",
              "parameters": [],
              "return_type": "Callable[[np.ndarray], np.ndarray]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "solve_transpose_vec",
              "parameters": [],
              "return_type": "Callable[[PETSc.Vec], PETSc.Vec]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "solve_block",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rhs_block",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "refactorize",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "operator",
                  "param_type": "sp.spmatrix | PETSc.Mat"
                }
              ],
              "return_type": "Factorization",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "destroy",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "_CudssShellPC",
              "parameters": [],
              "return_type": "_CudssShellPC",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "apply",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "pc",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "x",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "y",
                  "param_type": "Any"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_aij_pattern",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mat",
                  "param_type": "PETSc.Mat"
                }
              ],
              "return_type": "tuple[np.ndarray, np.ndarray]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_petsc",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "factorize.py",
          "responsibilities": [
            "Build and own reusable sparse factorizations",
            "Provide repeated forward and transpose solves",
            "Support block RHS solves and refactorization",
            "Integrate MUMPS/PETSc direct and iterative policies",
            "Bridge cuDSS solvers into PETSc PC shells"
          ],
          "source_summary": "Defines Factorization with numpy and PETSc Vec solve paths, transpose callables, block RHS solves, and refactorize/destroy lifecycle methods. Includes helpers for AIJ pattern extraction, local numpy-to-Vec conversion, lazy PETSc import, and _CudssShellPC to apply CudssSolver inside PETSc preconditioning.",
          "summary": "Core reusable PETSc factorization layer offering factorize-once solve-many interfaces for shift-invert LNSE workflows."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils.petsc_core",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils.petsc_init",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing.TYPE_CHECKING",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Small focused module that builds shifted operators (e.g., A - σB style forms) consumed by eigenvalue and resolvent pipelines. It bridges SciPy sparse matrices and PETSc matrices with careful type-only PETSc imports to avoid hard import-time coupling.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/linalg/shifted.py",
          "importance_score": 0.78,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "shifted_operator",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "shifted.py",
          "responsibilities": [
            "Build shifted operators for solvers",
            "Support SciPy sparse and PETSc matrix inputs",
            "Initialize SLEPc/PETSc context when needed",
            "Provide operator seams for shift-invert workflows"
          ],
          "source_summary": "Imports scipy.sparse and PETSc matrix detection/init helpers, then exposes shifted_operator construction logic for stability/resolvent use. TYPE_CHECKING guards keep PETSc.Mat annotations deferred while runtime paths handle both SciPy and PETSc operator kinds.",
          "summary": "Constructs shifted linear operators used by stability analysis and resolvent solvers."
        }
      ],
      "importance_score": 0.92,
      "key_files": [
        "factorize.py",
        "backends.py",
        "cudss_factor.py",
        "shifted.py",
        "__init__.py"
      ],
      "name": "linalg",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/linalg",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The linalg package provides reusable sparse-direct factorizations and shifted operators for LNSE stability and resolvent solvers in linstabpy. It unifies PETSc/MUMPS backend selection, optional NVIDIA cuDSS GPU factorizations, and a factorize-once solve-many API behind a lazy public surface. Together the modules form the numerical core that shift-invert and resolvent workflows depend on for exact and iterative linear solves."
    },
    {
      "file_count": 6,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "fdweights",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "sbp",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines the mesh package surface with lazy/getattr-style exports and an explicit __all__ list covering Mesh, geometry/operator views, SBP helpers, and filter builders. It documents the package as mesh infrastructure for SBP finite differences, structured grids, and integration weights. Callers typically import from this module rather than reaching into individual implementation files.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/mesh/__init__.py",
          "importance_score": 0.75,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "__getattr__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "__init__.py",
          "responsibilities": [
            "Expose the public mesh package API",
            "Re-export SBP and filter builders",
            "Document package purpose and license",
            "Support lazy attribute access via __getattr__"
          ],
          "source_summary": "Contains SPDX GPL-3.0 header, package docstring, imports from fdweights and sbp, and __all__ listing Mesh, MeshGeometry, MeshOperators, MeshRuntimeState, MeshFilters, build_mesh_filters, and SBP construction symbols. Implements __getattr__ for deferred attribute resolution of public names.",
          "summary": "Package entry point that re-exports mesh types, SBP builders, filters, and fdweights under a single public API."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "functools.cache",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy.typing",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements the classic Fornberg algorithm to obtain differentiation weights at a target point xi given irregular or regular nodes. Used as a low-level building block for interior SBP stencils and other FD constructions in the mesh stack. The module is small, cached where appropriate, and numerically focused.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/mesh/fdweights.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "fdweights",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "xi",
                  "param_type": "float"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "x",
                  "param_type": "ArrayLike"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "m",
                  "param_type": "int"
                }
              ],
              "return_type": "NDArray[np.float64]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "weight",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "q",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "j",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "k",
                  "param_type": "int"
                }
              ],
              "return_type": "float",
              "visibility": ""
            }
          ],
          "name": "fdweights.py",
          "responsibilities": [
            "Compute Fornberg FD weights for arbitrary nodes",
            "Support arbitrary derivative order m",
            "Handle edge cases such as empty node sets",
            "Cache intermediate weight computations"
          ],
          "source_summary": "Defines fdweights(xi, x, m) returning an NDArray of weights for derivative order m, with empty-input handling and numpy ArrayLike conversion. Includes an internal weight(q, j, k) helper used in the recursive Fornberg table construction, with functools.cache for efficiency.",
          "summary": "Computes Fornberg finite-difference weights for arbitrary node sets and derivative orders."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "dataclasses",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "collections.abc.Callable",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides MeshFilters and builder helpers that construct implicit filter systems on structured 1D lines and compose them into 2D separable operators used by Mesh. It deliberately avoids PETSc on this path and uses scipy.sparse plus callable solve/filter wrappers. This is essential for numerical stabilization of high-order SBP discretizations.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/mesh/filters.py",
          "importance_score": 0.85,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "MeshFilters",
              "parameters": [],
              "return_type": "MeshFilters",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_to_columns",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "v",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "tuple[np.ndarray, bool]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_from_columns",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "v2d",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "scalar_input",
                  "param_type": "bool"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "filter_fn",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "v",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "filter_ct_fn",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "v",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "identity",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rhs",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "solve",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rhs",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "solve_transpose",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rhs",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_is_no_filter",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "alpha_filter",
                  "param_type": "float | str"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            }
          ],
          "name": "filters.py",
          "responsibilities": [
            "Build implicit high-order spatial filters",
            "Package filter and transpose solve callables",
            "Support separable 2D filter operator pairs",
            "Detect no-filter configuration and identity paths"
          ],
          "source_summary": "Defines MeshFilters dataclass/container and helpers _to_columns/_from_columns for array layout, plus filter_fn, filter_ct_fn, identity, solve, and solve_transpose callables. Includes _is_no_filter for alpha_filter disable paths and sparse linear-algebra setup for implicit 10th-order filters translated from MATLAB.",
          "summary": "Builds implicit 10th-order spatial filters and separable 2D mesh filter operator pairs from MATLAB GetFilter/CreateFilter logic."
        },
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "dataclasses",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numbers",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "sbp",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "filters",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Owns the primary Mesh object and grouped views MeshGeometry, MeshOperators, and MeshRuntimeState so consumers can declare which contract slice they need. Implements creation, deformation (x-only, y-only, full 2D), pole-line handling, masking, and operator attachment that ties SBP matrices and filters into a mutable runtime carrier. This is the highest-value file in the directory and a core domain module for the solver stack.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/mesh/mesh.py",
          "importance_score": 0.98,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_validate_range_pair",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "tuple[float, float]"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "MeshGeometry",
              "parameters": [],
              "return_type": "MeshGeometry",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "MeshOperators",
              "parameters": [],
              "return_type": "MeshOperators",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "MeshRuntimeState",
              "parameters": [],
              "return_type": "MeshRuntimeState",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "Mesh",
              "parameters": [],
              "return_type": "Mesh",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "geometry",
              "parameters": [],
              "return_type": "MeshGeometry",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "operators",
              "parameters": [],
              "return_type": "MeshOperators",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "runtime_state",
              "parameters": [],
              "return_type": "MeshRuntimeState",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_deform_x_only",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "X_new",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "p",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_deform_y_only",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "Y_new",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "p",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_deform_full_2d",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "X_new",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "Y_new",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "p",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "mask",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mask_array",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "Mesh",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_starts_at_pole_line",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "y0",
                  "param_type": "float"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dy",
                  "param_type": "float"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            }
          ],
          "name": "mesh.py",
          "responsibilities": [
            "Construct and hold structured mesh runtime state",
            "Expose geometry and discrete operator views",
            "Support mesh deformation and masking",
            "Integrate SBP operators and integration weights",
            "Handle axisymmetric pole-line special cases"
          ],
          "source_summary": "Large module (~1767 lines) defining validation helpers, dataclasses/classes for geometry/operators/runtime state, and Mesh with properties geometry/operators/runtime_state. Includes _deform_x_only, _deform_y_only, _deform_full_2d, mask, and pole-start detection, translating CreateMesh/AddDW2mesh/DeformMesh workflows into Python.",
          "summary": "Central structured Mesh implementation with geometry, SBP operators, runtime state, deformation, and masking, ported from LinStab2D MATLAB."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Core numerical library for summation-by-parts operators following Strand and Mattsson & Nordström, plus Mohseni–Colonius pole matrices for axisymmetric grids. Builds differentiation and norm/dissipation-related sparse matrices consumed by Mesh. Without this module the structured-grid spatial discretization cannot be assembled.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/mesh/sbp.py",
          "importance_score": 0.93,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "sbp_stencils",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order",
                  "param_type": "int"
                }
              ],
              "return_type": "dict[str, np.ndarray]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "min_points",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order",
                  "param_type": "int"
                }
              ],
              "return_type": "int",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "build_sbp_matrix",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dz",
                  "param_type": "float"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order",
                  "param_type": "int"
                }
              ],
              "return_type": "tuple[sp.csr_matrix, sp.csr_matrix]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_build_sbp_with_pole",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dz",
                  "param_type": "float"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order",
                  "param_type": "int"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_mohseni_pole_stencil",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "a_interior",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "h",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_build_sbp_norm_diag",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "h",
                  "param_type": "float"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order_interior",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_vandermonde_centered_stencil",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "half_w",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "derivative_order",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_build_dp_matrix",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "h",
                  "param_type": "float"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "p",
                  "param_type": "int"
                }
              ],
              "return_type": "sp.csr_matrix",
              "visibility": ""
            }
          ],
          "name": "sbp.py",
          "responsibilities": [
            "Provide SBP stencil coefficients by order",
            "Assemble sparse SBP differentiation matrices",
            "Support minimum grid-point constraints",
            "Build pole-condition matrices for axisymmetric grids",
            "Construct norm and dissipation-related operators"
          ],
          "source_summary": "Exports sbp_stencils, min_points, and build_sbp_matrix returning CSR operator pairs; internals include pole-aware builders, Mohseni pole stencils, diagonal norm construction, Vandermonde-centered stencils, and dissipation matrix helpers. Documents orders 2/4/6/8 and literature references for coefficient provenance.",
          "summary": "Implements standard SBP finite-difference stencils and sparse matrix construction for orders 2–8, including pole treatments."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy.typing",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "fdweights",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Generates diagonal-norm SBP operators of interior order 2p with boundary closures of order p using Strand/Mattsson–Nordström constraints and Fornberg interior coefficients. Complements the hand-coded stencils in sbp.py by deriving coefficients from the SBP algebraic system. Useful for extending or verifying operator families without hard-coding every order.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/mesh/sbp_generator.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "fornberg_interior",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "deriv",
                  "param_type": "int"
                }
              ],
              "return_type": "NDArray[np.float64]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_sbp_norm_weights",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order",
                  "param_type": "int"
                }
              ],
              "return_type": "NDArray[np.float64]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_sbp_closure_width",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order",
                  "param_type": "int"
                }
              ],
              "return_type": "int",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_sbp_boundary_rows",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order",
                  "param_type": "int"
                }
              ],
              "return_type": "int",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_interior_coeff",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "a_interior",
                  "param_type": "NDArray[np.float64]"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "row",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "col",
                  "param_type": "int"
                }
              ],
              "return_type": "float",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_build_accuracy_eq",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "row_i",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "k",
                  "param_type": "int"
                }
              ],
              "return_type": "tuple[NDArray[np.float64], float]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "generate_sbp_stencils",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order",
                  "param_type": "int"
                }
              ],
              "return_type": "dict[str, NDArray[np.float64]]",
              "visibility": ""
            }
          ],
          "name": "sbp_generator.py",
          "responsibilities": [
            "Generate interior Fornberg FD coefficients",
            "Derive SBP boundary closure stencils",
            "Encode norm weights and closure widths by order",
            "Assemble constrained accuracy equations for SBP"
          ],
          "source_summary": "Provides fornberg_interior, norm-weight and closure-width helpers, accuracy-equation builders, and generate_sbp_stencils(order) returning coefficient dictionaries. Encodes Q+Q^T=B structure with diagonal H and boundary matrix B for consistent SBP property enforcement.",
          "summary": "Automatically generates SBP interior and boundary-closure stencils via Fornberg weights and constrained accuracy equations."
        }
      ],
      "importance_score": 0.92,
      "key_files": [
        "mesh.py",
        "sbp.py",
        "filters.py",
        "sbp_generator.py",
        "__init__.py"
      ],
      "name": "mesh",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/mesh",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The mesh package provides core numerical infrastructure for structured grids in linstabpy: SBP finite-difference operators, integration weights, spatial filters, and the mutable Mesh runtime object. It ports MATLAB LinStab2D mesh routines into a modular Python API where mesh.py orchestrates geometry/operators/state, sbp.py and sbp_generator.py supply differentiation matrices, filters.py builds implicit filters, and fdweights.py computes Fornberg weights. Together these files form the discretization backbone for linear stability analysis on structured and axisymmetric domains."
    },
    {
      "file_count": 6,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "assembly",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "coefficients",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "energy_norm",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "spatial",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines the operators package surface for LNSE operator construction, coefficients, assembly, and energy norms. It aggregates symbols from assembly, coefficients, energy_norm, and spatial so callers can import a stable public API without reaching into internal modules.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/operators/__init__.py",
          "importance_score": 0.75,
          "interfaces": [],
          "name": "__init__.py",
          "responsibilities": [
            "Expose public operators API",
            "Re-export assembly and coefficient builders",
            "Document package purpose"
          ],
          "source_summary": "A short GPL-licensed init module that documents LNSE operator construction and from-imports LNSESystem, build_lnse_operator variants, coefficient compute functions, compute_chu_norm, and spatial operator builders. No local logic beyond re-exports.",
          "summary": "Package entry that re-exports the public LNSE operator API: assembly builders, coefficient helpers, energy norm, and spatial operators."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "PETSc",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "coefficients",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "metrics",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Central implementation of temporal LNSE assembly, translating GetLinProblem.m into a structured pipeline with request objects, backend selection, sponge application, and finalization. It produces the 5*ngp operator for state q=[rho,u,v,w,T] and coordinates SciPy CSC and PETSc Mat paths with a fixed transformation order.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/operators/assembly.py",
          "importance_score": 0.98,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "LNSESystem",
              "parameters": [],
              "return_type": "LNSESystem",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "build_lnse_operator",
              "parameters": [],
              "return_type": "sp.csc_matrix",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "build_lnse_operator_petsc",
              "parameters": [],
              "return_type": "PETSc.Mat",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "build_lnse_system",
              "parameters": [],
              "return_type": "LNSESystem",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "build_lnse_system_result",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_decide_lnse_backend",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "request",
                  "param_type": "_LNSEAssemblyRequest"
                }
              ],
              "return_type": "_LNSEBackendDecision",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_build_lnse_system_from_request",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "request",
                  "param_type": "_LNSEAssemblyRequest"
                }
              ],
              "return_type": "LNSESystem",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_assemble_from_prepared",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "prep",
                  "param_type": "dict"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "floquet_exp",
                  "param_type": "float"
                }
              ],
              "return_type": "sp.csc_matrix",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "apply_sponge",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "L0",
                  "param_type": "sp.spmatrix"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "sponge_diag",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "sp.spmatrix",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "finalize",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "L0",
                  "param_type": "sp.spmatrix"
                }
              ],
              "return_type": "sp.csc_matrix",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_resolve_floquet_shift_mode",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mode",
                  "param_type": "str | None"
                }
              ],
              "return_type": "FloquetShiftMode",
              "visibility": ""
            }
          ],
          "name": "assembly.py",
          "responsibilities": [
            "Assemble temporal LNSE operators",
            "Select and run SciPy vs PETSc backends",
            "Apply sponge and finalize sparse matrices",
            "Manage Floquet shift and BC-related assembly metadata",
            "Expose LNSESystem and high-level build APIs"
          ],
          "source_summary": "Large module (~1946 lines) with LNSESystem, assembly request/decision types, SciPy and PETSc backends, Floquet shift handling, sponge application, and helpers to prepare coefficients and assemble from prepared data. Public builders include build_lnse_operator, build_lnse_operator_petsc, build_lnse_system, and related result constructors.",
          "summary": "Main LNSE operator assembly module that builds the large sparse L0 system from coefficients and FD matrices for SciPy and PETSc backends."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements the physics-side coefficient generation that assembly consumes, ported from getCoeffsLaminar_Cartesian.m and related MATLAB transforms. It handles coordinate-system and variable-ordering differences between MATLAB internal axes and physical streamwise/wall-normal/spanwise conventions.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/operators/coefficients.py",
          "importance_score": 0.93,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "LNSECoefficients",
              "parameters": [],
              "return_type": "LNSECoefficients",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "compute_cartesian_coefficients",
              "parameters": [],
              "return_type": "LNSECoefficients",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "compute_cylindrical_coefficients",
              "parameters": [],
              "return_type": "LNSECoefficients",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_PhysicalCoefficientBuilder.set",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source_name",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "row",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "col",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_PhysicalCoefficientBuilder.finish",
              "parameters": [],
              "return_type": "LNSECoefficients",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_expected_coefficient_keys",
              "parameters": [],
              "return_type": "list[str]",
              "visibility": ""
            }
          ],
          "name": "coefficients.py",
          "responsibilities": [
            "Build Cartesian LNSE coefficients",
            "Build cylindrical/axisymmetric coefficients",
            "Normalize coordinate and variable ordering",
            "Validate and package coefficient keys into LNSECoefficients"
          ],
          "source_summary": "Substantial module (~1781 lines) defining LNSECoefficients, a physical coefficient builder with set/finish, expected key validation, and compute_cartesian_coefficients / compute_cylindrical_coefficients entry points that fill block coefficient arrays used by operator assembly.",
          "summary": "Computes LNSE physical coefficient matrices for Cartesian and cylindrical compressible flows from baseflow fields."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides the energy inner-product weights used when analyzing compressible perturbations, translated from GetCompEnergyNorm.m. Supports Cartesian and axisymmetric models, with radial weighting by r in axisymmetric cases.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/operators/energy_norm.py",
          "importance_score": 0.72,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "compute_chu_norm",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "W_int",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "RHO",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "T",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kappa",
                  "param_type": "float"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            }
          ],
          "name": "energy_norm.py",
          "responsibilities": [
            "Compute Chu compressible energy norm weights",
            "Support Cartesian and axisymmetric formulations",
            "Incorporate radial integration factors when needed"
          ],
          "source_summary": "Compact module exposing compute_chu_norm(W_int, RHO, T, kappa, ...) that combines integration weights with thermodynamic baseflow fields to form Chu-norm diagonal or block weights for modal energy measures.",
          "summary": "Implements Chu's compressible energy norm for weighting perturbation energy in stability analysis."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "json",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "logging",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.config",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Supports observability and performance of the heavy assembly path without owning physics. Collectors record stages and metadata, optionally write JSON summaries, and provide a CSR derivative cache used while assembling sparse operators.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/operators/metrics.py",
          "importance_score": 0.55,
          "interfaces": [
            {
              "description": null,
              "interface_type": "method",
              "name": "_AssemblyMetricsCollector.record_stage",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "payload",
                  "param_type": "dict[str, Any]"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_AssemblyMetricsCollector.update_metadata",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "overwrite",
                  "param_type": "bool"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kwargs",
                  "param_type": "dict"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_AssemblyMetricsCollector.finish",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_resolve_assembly_metrics_config",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "solver_cfg",
                  "param_type": "dict | None"
                }
              ],
              "return_type": "_AssemblyMetricsConfig",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_assembly_stage",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "Iterator[dict[str, Any]]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "_CSRDerivativeCache",
              "parameters": [],
              "return_type": "_CSRDerivativeCache",
              "visibility": ""
            }
          ],
          "name": "metrics.py",
          "responsibilities": [
            "Collect assembly stage timings and metadata",
            "Emit metric summaries to logs or files",
            "Resolve metrics config from solver settings",
            "Cache CSR derivative structures for reuse"
          ],
          "source_summary": "Defines _AssemblyMetricsConfig/Collector, context helpers (_assembly_stage, _note_assembly_metadata), rank/size detection, JSON serialization of metric values, and _CSRDerivativeCache. Uses contextvars, logging, pathlib, numpy, and scipy.sparse.",
          "summary": "Instrumentation for assembly timing, stage metrics, JSON logging, and CSR derivative caching during operator builds."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "BaseFlow",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "Mesh",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "assembly",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Extends the temporal assembly machinery to spatial eigenproblems used in LNSE spatial stability analysis. Reuses preparation and index helpers from assembly and builds block operators, including PETSc block-matrix assembly.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/operators/spatial.py",
          "importance_score": 0.88,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "build_spatial_operator",
              "parameters": [],
              "return_type": "sp.spmatrix",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "build_spatial_operator_petsc",
              "parameters": [],
              "return_type": "PETSc.Mat",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_assemble_block_matrix_petsc",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "blocks",
                  "param_type": "list[list[PETSc.Mat | None]]"
                }
              ],
              "return_type": "PETSc.Mat",
              "visibility": ""
            }
          ],
          "name": "spatial.py",
          "responsibilities": [
            "Build spatial LNSE eigenproblem operators",
            "Provide SciPy and PETSc spatial assembly paths",
            "Assemble PETSc block matrices from coefficient blocks",
            "Reuse shared coefficient preparation and validation"
          ],
          "source_summary": "Module (~279 lines) importing BaseFlow, Mesh, and private assembly helpers; exposes build_spatial_operator and build_spatial_operator_petsc plus _assemble_block_matrix_petsc to construct the spatial problem matrices from prepared coefficient structures.",
          "summary": "Assembles spatial stability eigenproblem operators from prepared LNSE coefficients for SciPy and PETSc."
        }
      ],
      "importance_score": 0.95,
      "key_files": [
        "assembly.py",
        "coefficients.py",
        "spatial.py",
        "energy_norm.py",
        "__init__.py"
      ],
      "name": "operators",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/operators",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The operators package is the core numerical engine of linstabpy for constructing linearized Navier–Stokes (LNSE) operators used in temporal and spatial stability analysis. It translates MATLAB LinStab2D logic into Python: coefficient generation for Cartesian/cylindrical flows, sparse/PETSc operator assembly, spatial eigenproblem construction, Chu energy norms, and assembly metrics. Together these modules form the primary scientific backend that turns baseflow and mesh data into discrete stability operators."
    },
    {
      "file_count": 5,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.resolvent.operator",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.resolvent.gain",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.resolvent.analysis",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.resolvent.conservative_forcing",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines the public surface of linstabpy.resolvent via lazy __getattr__/__dir__ so heavy PETSc/SLEPc imports stay deferred. It maps names such as ResolventOperator, WeightedNormalOperator, and resolvent_analysis to submodule paths.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/resolvent/__init__.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "__getattr__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__dir__",
              "parameters": [],
              "return_type": "list[str]",
              "visibility": ""
            }
          ],
          "name": "__init__.py",
          "responsibilities": [
            "Lazy public API export",
            "Deferred submodule loading",
            "Package-level discoverability via __dir__"
          ],
          "source_summary": "Holds a _LAZY_EXPORTS dict linking public symbols to submodules and implements __getattr__ and __dir__ for deferred loading. Keeps the package import PETSc-light at startup while exposing the main resolvent utilities.",
          "summary": "Package entry that lazily re-exports resolvent operator, gain, analysis, and conservative-forcing APIs."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "petsc4py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "slepc4py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.resolvent.operator",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.resolvent.gain",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements resolvent analysis on H = C @ inv(L0 - i*omega*I) @ B by solving the weighted eigenproblem for H^H W H via SLEPc EPS on a MatShell. Provides validation, residual norms, and the main analysis workflow aligned with a MATLAB reference path.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/resolvent/analysis.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_operator_ndof",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "operator",
                  "param_type": "sp.spmatrix | PETSc.Mat"
                }
              ],
              "return_type": "int",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_validate_finite_number",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "float | complex"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_w_norm",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "residual",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "float",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_l2_norm",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "residual",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "float",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "resolvent_analysis",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "analysis.py",
          "responsibilities": [
            "Run frequency-domain resolvent analysis",
            "Validate operator and numeric inputs",
            "Compute weighted and L2 residual norms",
            "Drive SLEPc eigenproblems on shell matrices",
            "Return optimal gains and forcing/response pairs"
          ],
          "source_summary": "Contains helpers such as _operator_ndof, _validate_finite_number, and weighted/L2 residual norms, plus the bulk resolvent_analysis logic. Orchestrates operator application, weighting, and singular/eigenmode extraction for frequency-domain optimal forcing and response.",
          "summary": "Primary resolvent analysis driver computing optimal forcing/response pairs and gains with PETSc/SLEPc."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Supports physically motivated forcing on (rho u)', (rho v)', (rho w)' instead of the default five primitive perturbation variables. Builds coefficient/matrices and dewhitening helpers so optimisation variables match shear-layer/shock-interaction physics.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/resolvent/conservative_forcing.py",
          "importance_score": 0.82,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "validate_conservative_momentum_vars",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "raw_vars",
                  "param_type": "object"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "tuple[str, ...]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "build_conservative_momentum_matrix",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "conservative_momentum_coefficients",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "dewhiten_forcing",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "conservative_forcing.py",
          "responsibilities": [
            "Validate conservative momentum variable sets",
            "Build conservative-momentum forcing matrices",
            "Provide coefficient transforms for forcing space",
            "Dewhiten forcing back to physical variables"
          ],
          "source_summary": "Exposes validation of conservative momentum variable names and builders such as build_conservative_momentum_matrix, conservative_momentum_coefficients, and dewhiten_forcing. Bridges region masks and resolvent maps from primitive space into conservative momentum forcing space.",
          "summary": "Maps resolvent forcing between primitive variables and conservative momentum equations."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "logging",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils.petsc_init",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "petsc4py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "slepc4py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines WeightedNormalOperator for matvecs of the form invW * H^H * W * H(v) as a PETSc MatShell. Handles SLEPc/MPI init ordering constraints and NCV sizing for mode extraction.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/resolvent/gain.py",
          "importance_score": 0.88,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "WeightedNormalOperator",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "mult",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mat",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "x",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "y",
                  "param_type": "Any"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "create_shell",
              "parameters": [],
              "return_type": "PETSc.Mat",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_resolve_resolvent_ncv",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ndof",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n_modes",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "requested_ncv",
                  "param_type": "int | None"
                }
              ],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "gain.py",
          "responsibilities": [
            "Represent weighted H^H W H operator",
            "Provide PETSc MatShell mult implementation",
            "Create shell matrices for SLEPc EPS",
            "Choose appropriate NCV for mode counts",
            "Ensure correct MPI/SLEPc init order"
          ],
          "source_summary": "Creates shell matrices with mult callbacks, resolves Krylov subspace dimension via _resolve_resolvent_ncv, and coordinates PETSc/SLEPc initialization. Central to computing singular values/gains of the weighted resolvent.",
          "summary": "Implements the weighted normal operator shell used to extract resolvent gains with SLEPc."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils.petsc_core",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "petsc4py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "ResolventOperator applies H and H^H using linear solves against (L0 - i*omega*I) with input/output maps B and C. Uses PETSc-aware vector helpers while keeping type-only PETSc imports so package import stays light.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/resolvent/operator.py",
          "importance_score": 0.9,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "ResolventOperator",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "apply",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "v",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "apply_adjoint",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "v",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_probe",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rng",
                  "param_type": "np.random.RandomState"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            }
          ],
          "name": "operator.py",
          "responsibilities": [
            "Apply resolvent operator H to vectors",
            "Apply adjoint resolvent actions",
            "Probe operator behavior with random vectors",
            "Bridge NumPy and PETSc vector representations"
          ],
          "source_summary": "Implements apply, apply_adjoint, and random _probe utilities over sparse or PETSc matrices. Converts between NumPy and PETSc vectors via petsc_core helpers for consistent operator actions in analysis and gain modules.",
          "summary": "Encapsulates forward and adjoint actions of the resolvent operator H."
        }
      ],
      "importance_score": 0.92,
      "key_files": [
        "analysis.py",
        "operator.py",
        "gain.py",
        "conservative_forcing.py",
        "__init__.py"
      ],
      "name": "resolvent",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/resolvent",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The resolvent package implements core resolvent analysis for linearised Navier–Stokes: operator application, weighted gain eigenproblems via PETSc/SLEPc, and conservative-momentum forcing transforms. Files collaborate through lazy package exports, operator and gain shells feed analysis, while conservative_forcing adapts the forcing space for shear-layer/shock problems."
    },
    {
      "file_count": 5,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.resolvent.analysis",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines the public surface of linstabpy.solvers with deferred attribute resolution so import cost stays low and SLEPc init remains submodule-owned. It documents the lazy contract with the top-level package and maps export names to implementing modules.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/__init__.py",
          "importance_score": 0.75,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "__getattr__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__dir__",
              "parameters": [],
              "return_type": "list",
              "visibility": ""
            }
          ],
          "name": "__init__.py",
          "responsibilities": [
            "Lazy public API exports",
            "Avoid eager PETSc/SLEPc import",
            "Expose package __dir__ for discovery"
          ],
          "source_summary": "Contains _LAZY_EXPORTS mapping names such as resolvent_analysis to target modules, plus __getattr__ and __dir__ for PEP 562 lazy loading. Docstring explains why eager re-exports would defeat lazy PETSc/SLEPc initialization.",
          "summary": "Package entry that lazily re-exports solver and resolvent symbols via PEP 562 without importing petsc4py/slepc4py at load time."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils._solver_cfg.SolverConfig",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils.petsc_init.init_slepc_from_argv",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Centralizes EPS-related plumbing so eigenvalue and related solvers share consistent convergence/error resolution and monitor formatting. Initializes SLEPc from argv and coordinates MPI threading expectations with mpi4py before PETSc binding.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/_eps_helpers.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_resolve_eps_convergence_test",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "SLEPc.EPS.Conv",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_resolve_eps_error_type",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "SLEPc.EPS.ErrorType",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_monitor_state_details",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "state",
                  "param_type": "dict[str, float | int] | None"
                }
              ],
              "return_type": "list[str]",
              "visibility": ""
            }
          ],
          "name": "_eps_helpers.py",
          "responsibilities": [
            "Map config names to SLEPc EPS enums",
            "Format EPS monitor state details",
            "Initialize SLEPc safely with MPI constraints",
            "Share EPS helpers across solvers"
          ],
          "source_summary": "Imports SolverConfig and init_slepc_from_argv, then defines helpers such as _resolve_eps_convergence_test, _resolve_eps_error_type, and _monitor_state_details. Comments document MPI_THREAD_MULTIPLE vs FUNNELED ordering and factorization backend constraints.",
          "summary": "Shared SLEPc EPS helper utilities for convergence tests, error types, monitoring, and solver config used across temporal, adjoint, and resolvent paths."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "PETSc",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "SLEPc",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.solvers._eps_helpers",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements the main eigenanalysis service driven by solver.eigen YAML configuration, including direct shift-invert paths and factor backend selection (MUMPS/LU and env overrides). Provides KSP option application, residual reordering, safe PETSc info access, and a dedicated error for STRUMPACK factorization failures.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/eigenvalue.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "StrumpackFactorizationFailed",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "__init__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "message",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "factor_backend",
                  "param_type": "str"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_apply_ksp_options",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ksp",
                  "param_type": "PETSc.KSP"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "dict | None"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_nearest_order",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "values",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "target",
                  "param_type": "complex"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_safe_petsc_info",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "callable_obj",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "default",
                  "param_type": "object"
                }
              ],
              "return_type": "object",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_reorder_info_residuals",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "info",
                  "param_type": "dict[str, object]"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "order",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "apply",
              "parameters": [],
              "return_type": "dict[str, object]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_resolve_st_factor_backend",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "st_solver",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ecfg",
                  "param_type": "dict | None"
                }
              ],
              "return_type": "str | None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_probe_residual_rel_max",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ecfg",
                  "param_type": "dict | None"
                }
              ],
              "return_type": "float",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__getattr__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__dir__",
              "parameters": [],
              "return_type": "list[str]",
              "visibility": ""
            }
          ],
          "name": "eigenvalue.py",
          "responsibilities": [
            "Run temporal and adjoint EPS solves",
            "Configure shift-invert and LU factor backends",
            "Apply KSP/PETSc solver options",
            "Compute and reorder residuals and eigen-info",
            "Surface factorization failures clearly"
          ],
          "source_summary": "Defines StrumpackFactorizationFailed and many helpers such as _apply_ksp_options, _nearest_order, _reorder_info_residuals, _resolve_st_factor_backend, and _probe_residual_rel_max, plus apply entry points returning solver info dicts. Also uses lazy __getattr__/__dir__ patterns and extensive PETSc/SLEPc configuration for the reference direct path.",
          "summary": "Primary temporal and adjoint eigenvalue solver implementation using SLEPc EPS with shift-invert, factorization backends, and residual handling."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils.petsc_core",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Preserves historical import paths for solver modules while keeping canonical PETSc helpers in utils.petsc_core. Reduces churn for callers that still import conversion and shape helpers from solvers.petsc_utils.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/petsc_utils.py",
          "importance_score": 0.45,
          "interfaces": [],
          "name": "petsc_utils.py",
          "responsibilities": [
            "Maintain import compatibility",
            "Re-export PETSc core utilities",
            "Decouple solvers from utils relocation"
          ],
          "source_summary": "Re-exports symbols including _is_petsc_matrix, _matrix_nnz, _operator_shape, numpy_columns_to_petsc_vecs, petsc_to_scipy, petsc_vec, require_complex, scipy_to_petsc, and scipy_to_petsc_distributed. No local logic beyond import passthrough.",
          "summary": "Thin backward-compatible re-export layer for PETSc matrix/vector conversion utilities defined in linstabpy.utils.petsc_core."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "PETSc",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "SLEPc",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.solvers._eps_helpers",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils.petsc_init",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements spatial stability eigenproblems, falling back to solver.eigen settings for compatibility when spatial-specific YAML is absent. Handles direct shift-invert backend selection and complex PETSc LU probing similar to the temporal path.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/spatial_eig.py",
          "importance_score": 0.88,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "__getattr__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__dir__",
              "parameters": [],
              "return_type": "list[str]",
              "visibility": ""
            }
          ],
          "name": "spatial_eig.py",
          "responsibilities": [
            "Solve spatial generalized eigenproblems",
            "Resolve spatial vs eigen solver config",
            "Select PETSc LU factor backends",
            "Lazy-export spatial solver symbols"
          ],
          "source_summary": "Documents YAML-driven configuration, factor_backend precedence (YAML, LINSTABPY_FACTOR_BACKEND, mumps default), and complex LU probes. Exposes lazy __getattr__/__dir__ and imports SLEPc/PETSc-related solver infrastructure for generalized spatial EPS solves.",
          "summary": "Generalized SLEPc EPS eigensolvers for spatial stability analysis with solver.spatial config and factor_backend resolution."
        }
      ],
      "importance_score": 0.92,
      "key_files": [
        "eigenvalue.py",
        "spatial_eig.py",
        "_eps_helpers.py",
        "__init__.py",
        "petsc_utils.py"
      ],
      "name": "solvers",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers",
      "purpose": "other",
      "subdirectory_count": 1,
      "summary": "The solvers package implements core numerical eigenvalue and resolvent analysis for linstabpy using SLEPc/PETSc. It exposes temporal, adjoint, and spatial eigensolvers behind lazy imports so heavy MPI/PETSc dependencies load only when needed, with shared EPS helpers and PETSc utility re-exports supporting configuration, factorization backends, and residual monitoring."
    },
    {
      "file_count": 11,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "anchors",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "band_marking",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "break_even",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "certification",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "recycling",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Lazy public surface for the router package. It re-exports anchors, band marking, break-even routing, certification, and recycling APIs so callers import from one module.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/__init__.py",
          "importance_score": 0.55,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "__getattr__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__dir__",
              "parameters": [],
              "return_type": "list[str]",
              "visibility": ""
            }
          ],
          "name": "__init__.py",
          "responsibilities": [
            "Re-export public router APIs",
            "Provide lazy attribute access",
            "Document package purpose"
          ],
          "source_summary": "GPL-licensed package init with __getattr__ and __dir__ plus from-imports of public classes and functions from sibling modules.",
          "summary": "Package facade that re-exports resolvent sweep routing helpers and public types."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.solvers.router.band_marking",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils._solver_cfg",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Chooses factorization anchors on the omega grid using cheap probes, band distance, and config thresholds. Records why each anchor was triggered so later routing can reuse those decisions.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/anchors.py",
          "importance_score": 0.82,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "IterProbeResult",
              "parameters": [],
              "return_type": "IterProbeResult",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "AnchorTriggerRecord",
              "parameters": [],
              "return_type": "AnchorTriggerRecord",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "choose_anchors_adaptive",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "omega_grid",
                  "param_type": "Array"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "Cfg"
                }
              ],
              "return_type": "Array",
              "visibility": ""
            }
          ],
          "name": "anchors.py",
          "responsibilities": [
            "Place adaptive factorization anchors",
            "Probe iteration cost at sample frequencies",
            "Respect unstable-band distance",
            "Validate config numeric thresholds"
          ],
          "source_summary": "Defines IterProbeResult and AnchorTriggerRecord plus helpers to sample omegas, sort unique complex values, compute EVP radius, and pick nearest anchors from BandMarks and solver config.",
          "summary": "Adaptive pre-sweep anchor placement for stable-regime resolvent frequency sweeps."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Converts eigenvalue poles into danger radii and labels frequency samples as unstable, smooth, or ordinary. Downstream walkers use BandMarks to avoid cheap iterative solves near singularities.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/band_marking.py",
          "importance_score": 0.78,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "BandMarks",
              "parameters": [],
              "return_type": "BandMarks",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "label_band",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "omega",
                  "param_type": "complex"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "marks",
                  "param_type": "BandMarks"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "mark_unstable_band",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "omegas",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "Mapping[str, Any]"
                }
              ],
              "return_type": "BandMarks",
              "visibility": ""
            }
          ],
          "name": "band_marking.py",
          "responsibilities": [
            "Compute pole-distance band marks",
            "Label frequency samples",
            "Derive danger radius from grid spacing",
            "Read band-marking config"
          ],
          "source_summary": "Defines BandMarks and helpers for sweep spacing, danger radius, smooth regions, and config lookup. Documents the omega_pole mapping from temporal EVP values.",
          "summary": "Marks resolvent forcing-frequency bands from temporal EVP distance to poles."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Measures factorization and apply times on the operator pattern and computes the frequency-count break-even N_f*. Returns a RoutingDecision so the map can pick direct LU versus iterative sweep.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/break_even.py",
          "importance_score": 0.76,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "BreakEvenProbeResult",
              "parameters": [],
              "return_type": "BreakEvenProbeResult",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "RoutingDecision",
              "parameters": [],
              "return_type": "RoutingDecision",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "break_even_decision",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "probe",
                  "param_type": "BreakEvenProbeResult"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "Mapping[str, Any]"
                }
              ],
              "return_type": "RoutingDecision",
              "visibility": ""
            }
          ],
          "name": "break_even.py",
          "responsibilities": [
            "Measure factorization versus apply cost",
            "Compute break-even frequency count",
            "Emit routing decisions",
            "Validate probe measurements"
          ],
          "source_summary": "Defines BreakEvenProbeResult and RoutingDecision with config helpers, timers, representative RHS construction, and validation of measured probe costs using SciPy splu.",
          "summary": "Compares direct sparse factorizations against anchor-FGMRES sweep cost."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Independent of the sweep router. Verifies adjoint consistency and related RSVD assumptions, then can run a small dense Rayleigh-Ritz SVD through certified_rsvd.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/certification.py",
          "importance_score": 0.74,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "CheckResult",
              "parameters": [],
              "return_type": "CheckResult",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "CertificationReport",
              "parameters": [],
              "return_type": "CertificationReport",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "certified_rsvd",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "apply_H",
                  "param_type": "Callable"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "apply_H_adj",
                  "param_type": "Callable"
                }
              ],
              "return_type": "CertificationReport",
              "visibility": ""
            }
          ],
          "name": "certification.py",
          "responsibilities": [
            "Certify resolvent apply/adjoint seams",
            "Aggregate check reports",
            "Run certified RSVD after checks",
            "Expose serializable check results"
          ],
          "source_summary": "Defines CheckResult and CertificationReport with as_dict/passed helpers plus vector-norm utilities used by adjoint and action checks.",
          "summary": "Algebraic certification checks for inexact resolvent actions before RSVD."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "csv",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Writes each finished map point immediately so long jobs can resume. Also tracks last factorization seconds and remaining wall time for factor decisions.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/map_record.py",
          "importance_score": 0.58,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "append_row",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "Path"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "fields",
                  "param_type": "Sequence[str]"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "row",
                  "param_type": "Sequence[object]"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "atomic_savez",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "Path"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "payload",
                  "param_type": "dict"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "last_factor_seconds",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "Path"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "field",
                  "param_type": "str"
                }
              ],
              "return_type": "float",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "job_remaining_s",
              "parameters": [],
              "return_type": "float | None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "time_left_for_factor",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "remaining_s",
                  "param_type": "float | None"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "need_s",
                  "param_type": "float"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            }
          ],
          "name": "map_record.py",
          "responsibilities": [
            "Append flushed CSV map rows",
            "Atomically save numpy payloads",
            "Read last factorization timing",
            "Gate factors on remaining job time"
          ],
          "source_summary": "CSV append with header-on-create, atomic np.savez, last_factor_seconds lookup, and job_remaining_s / time_left_for_factor helpers.",
          "summary": "Incremental map-point persistence and job-time remaining helpers."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "First frequencies pay a real factor and measure cost. Later points reuse that factor as a right preconditioner until a residual gate forces a new factor. LineWalker records solves and can abort or defer a walk.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/map_walk.py",
          "importance_score": 0.88,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "LineWalker",
              "parameters": [],
              "return_type": "LineWalker",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "WalkPointResult",
              "parameters": [],
              "return_type": "WalkPointResult",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "probe_right_hand_side",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "last_solve",
                  "param_type": "np.ndarray | None"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            }
          ],
          "name": "map_walk.py",
          "responsibilities": [
            "Walk frequency lines with reuse",
            "Measure factor wall and apply count",
            "Gate reuse with residual probes",
            "Record solve and transpose actions"
          ],
          "source_summary": "Defines WalkDeferred, WalkAborted, WalkPointResult, LineWalker, and gated/recording solve wrappers around LU factor replacement and GMRES probes.",
          "summary": "Walks a frequency line with measured MUMPS anchors and a residual gate."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Does not trust formula-only fill estimates. Probes the nonzero pattern, measures LU fill and bytes, then chooses rank and precision candidates that fit available RAM.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/ram_autoselect.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "SymbolicProbeResult",
              "parameters": [],
              "return_type": "SymbolicProbeResult",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "OperatingPoint",
              "parameters": [],
              "return_type": "OperatingPoint",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "available_ram_bytes",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "Mapping[str, Any] | object | None"
                }
              ],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "ram_autoselect.py",
          "responsibilities": [
            "Probe LU fill on the operator pattern",
            "Estimate factor memory from measurements",
            "Select rank and precision operating points",
            "Query available RAM"
          ],
          "source_summary": "Defines SymbolicProbeResult and OperatingPoint plus RAM discovery, CSC pattern helpers, well-conditioned probe matrices, and rank/precision candidate generation.",
          "summary": "RAM-calibrated operating-point selection using empirical sparse-LU probes."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements right-preconditioned GMRES with a subspace prepared outside each RSVD pass. The subspace is never adapted inside a solve so the linear map stays certifiable.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/recycling.py",
          "importance_score": 0.77,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "resolve_gmres_lengths",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "maxiter",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "restart",
                  "param_type": "int | None"
                }
              ],
              "return_type": "tuple[int, int]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "PreparedRecycleSubspace",
              "parameters": [],
              "return_type": "PreparedRecycleSubspace",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "empty_recycle_basis",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n",
                  "param_type": "int"
                }
              ],
              "return_type": "Array",
              "visibility": ""
            }
          ],
          "name": "recycling.py",
          "responsibilities": [
            "Prepare fixed recycle subspaces",
            "Run recycled GMRES solves",
            "Resolve restart and maxiter lengths",
            "Keep the certified linear map unchanged"
          ],
          "source_summary": "Defines PreparedRecycleSubspace, RecycledGmresResult, resolve_gmres_lengths, empty_recycle_basis, and basis-normalization helpers.",
          "summary": "Fixed-subspace recycled GMRES core for stable resolvent sweeps."
        },
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.solvers.router.band_marking",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Assembles one LNSE operator per beta, marks frequency bands, and walks each line with factor reuse and residual fallback. Leading gain still comes from SLEPc. This is the largest hub in the package.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/resolvent_map.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "RouterResult",
              "parameters": [],
              "return_type": "RouterResult",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_mark_unstable_band",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "L0_beta",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "f_range",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "Mapping[str, Any]"
                }
              ],
              "return_type": "BandMarks",
              "visibility": ""
            }
          ],
          "name": "resolvent_map.py",
          "responsibilities": [
            "Assemble LNSE operators per beta",
            "Orchestrate frequency-line walks",
            "Apply band marking and routing policy",
            "Collect gain-map router results"
          ],
          "source_summary": "Defines RouterResult and private assembly/direct-point types plus config, mesh, BC mask, heartbeat, and unstable-band helpers that orchestrate the full map.",
          "summary": "Turnkey beta/frequency resolvent gain-map orchestration entry point."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.linalg.backends",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.linalg.factorize",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Factors selected anchors then applies them as preconditioners across nearby frequencies with PETSc/SciPy backends. Records factor time, memory, and compression samples for routing feedback.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router/stable_sweep.py",
          "importance_score": 0.93,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "SweepResult",
              "parameters": [],
              "return_type": "SweepResult",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "resolve_gmres_lengths",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "maxiter",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "restart",
                  "param_type": "int | None"
                }
              ],
              "return_type": "tuple[int, int]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "record_compression_sample",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "operator",
                  "param_type": "sp.spmatrix | PETSc.Mat"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "factor",
                  "param_type": "Any"
                }
              ],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "stable_sweep.py",
          "responsibilities": [
            "Run anchor-preconditioned frequency sweeps",
            "Track factor time and memory",
            "Drive GMRES iteration controls",
            "Interface SciPy and PETSc operators"
          ],
          "source_summary": "Defines SweepResult plus GMRES controls, PETSc option scoping, matvec/factor metrics, nearest-anchor indexing, and compression sampling around sparse or PETSc operators.",
          "summary": "Anchor-preconditioned stable-regime resolvent frequency sweep engine."
        }
      ],
      "importance_score": 0.92,
      "key_files": [
        "resolvent_map.py",
        "stable_sweep.py",
        "map_walk.py",
        "anchors.py",
        "ram_autoselect.py"
      ],
      "name": "router",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/solvers/router",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The router package is the core orchestration layer for resolvent gain-map sweeps in linstabpy. It chooses factorization versus iterative strategies, places anchors, marks unstable frequency bands, certifies RSVD actions, and walks beta/frequency lines with RAM-aware operating points. Files form a pipeline from package exports through helpers into resolvent_map and stable_sweep entry points."
    },
    {
      "file_count": 4,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "functools.wraps",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing.Any",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.stability._eigenvalues",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines the public surface of linstabpy.stability via a lazy-export map and __getattr__/__dir__ hooks. It keeps import cost low while advertising eigenvalues, adjoint_eigenvalues, spatial_eigenvalues, and internal near-target solve helpers.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/stability/__init__.py",
          "importance_score": 0.78,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "__getattr__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "__dir__",
              "parameters": [],
              "return_type": "list[str]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_resolve",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_wrapper",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "args",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kwargs",
                  "param_type": "dict"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "__init__.py",
          "responsibilities": [
            "Define public package exports",
            "Lazy-load eigenvalue modules on attribute access",
            "Expose __dir__/__all__ for discoverability",
            "Avoid eager PETSc/SLEPc import cost"
          ],
          "source_summary": "Builds _LAZY_EXPORTS mapping public names to linstabpy.stability._eigenvalues, sorts non-private names into __all__, and implements __getattr__ resolution with optional wrappers. Uses functools.wraps and typing.Any for deferred attribute access.",
          "summary": "Package entry that lazily re-exports the main stability eigenvalue APIs so heavy SLEPc/PETSc imports load only on first use."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "dataclasses",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "logging",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy.utils.petsc_init",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Contains the bulk of the package's computational logic: configuring direct solvers (e.g. MUMPS), validating modes/weights, and driving shift-invert eigenvalue attempts. It initializes SLEPc from argv with careful MPI threading order and returns eigenvalues, eigenvectors, and solver metadata.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/stability/_eigenvalues.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_validate_finite_number",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "float | complex"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_validate_n_modes",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n_modes",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ndof",
                  "param_type": "int"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_validate_weight_vector",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ndof",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_mumps_direct_solver_cfg",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "solver_cfg",
                  "param_type": "dict | None"
                }
              ],
              "return_type": "dict",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_attempt",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "scfg",
                  "param_type": "dict | None"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "c",
                  "param_type": "Config"
                }
              ],
              "return_type": "tuple[np.ndarray, np.ndarray, dict]",
              "visibility": ""
            }
          ],
          "name": "_eigenvalues.py",
          "responsibilities": [
            "Solve temporal/adjoint/spatial eigenvalue problems",
            "Validate numeric inputs, mode counts, and weight vectors",
            "Configure sparse direct solvers for shift-invert",
            "Initialize SLEPc/MPI safely and report solve metadata"
          ],
          "source_summary": "Imports numpy, scipy.sparse, logging, dataclasses, and petsc/slepc init utilities. Provides validators (_validate_finite_number, _validate_n_modes, _validate_weight_vector), solver config helpers (_mumps_direct_solver_cfg), and _attempt-style solve paths that return mode arrays and diagnostics for near-target spectral analysis.",
          "summary": "Primary stability eigenvalue entry points implementing temporal, adjoint, and spatial eigen solves with SLEPc/PETSc and validation helpers."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements numerical acceptance criteria used after eigen solves to decide whether modes are trustworthy. It accounts for conjugate symmetry of linearized Navier–Stokes operators on real baseflows when measuring distance to the shift-invert target.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/stability/acceptance.py",
          "importance_score": 0.82,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_postcheck_target_distance",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "omega",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "target",
                  "param_type": "complex"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "conjugate_aware",
                  "param_type": "bool"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            }
          ],
          "name": "acceptance.py",
          "responsibilities": [
            "Compute per-mode distance to shift-invert target",
            "Apply conjugate-aware acceptance for real operators",
            "Support residual-based postchecks of eigenpairs",
            "Gate accepted vs rejected stability modes"
          ],
          "source_summary": "Defines helpers such as _postcheck_target_distance over omega arrays and a complex target, with optional conjugate-aware distance. Relies on numpy and documents residual/postcheck rationale for stability spectra quality control.",
          "summary": "Acceptance gates that score eigenvalue quality via target distance and residual postchecks, including conjugate-aware metrics for real baseflows."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "logging",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing.Any",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides validation and geometry helpers to reshape modes, eigenvalues, and weight vectors, then compute weighted norms used in wavemaker/structural-sensitivity maps. It sits downstream of eigenvalue solves as analysis tooling rather than the solver itself.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/stability/sensitivity.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_as_modes",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "modes",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_as_eigs",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "eigs",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n_modes",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_as_weight",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "weight",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ndof",
                  "param_type": "int"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_validate_grid_shape",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "grid_shape",
                  "param_type": "tuple[int, int]"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ndof",
                  "param_type": "int"
                }
              ],
              "return_type": "tuple[int, int, int]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_weighted_norm",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "vector",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "weight_diag",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "float",
              "visibility": ""
            }
          ],
          "name": "sensitivity.py",
          "responsibilities": [
            "Normalize and validate mode/eigenvalue arrays",
            "Validate discrete weight vectors and grid shapes",
            "Compute weighted vector norms for sensitivity",
            "Support structural-sensitivity post-processing workflows"
          ],
          "source_summary": "Implements _as_modes, _as_eigs, _as_weight, _validate_grid_shape, and _weighted_norm with numpy arrays and logging. Enforces 1-D/2-D mode layouts, eigenvalue counts, and grid shape consistency against ndof before sensitivity calculations.",
          "summary": "Structural-sensitivity post-processing utilities that combine direct/adjoint modes with weighted norms on the discrete grid."
        }
      ],
      "importance_score": 0.92,
      "key_files": [
        "_eigenvalues.py",
        "__init__.py",
        "acceptance.py",
        "sensitivity.py"
      ],
      "name": "stability",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/stability",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The stability package is the core domain module of linstabpy for linear stability analysis of fluid flows. It exposes lazy-loaded eigenvalue solvers (temporal, adjoint, spatial), acceptance/postcheck gates for computed spectra, and structural-sensitivity post-processing that ties direct and adjoint modes together. Together these files form the primary computational API for shift-invert eigenproblems via SLEPc/PETSc and related validation utilities."
    },
    {
      "file_count": 8,
      "file_insights": [
        {
          "code_purpose": "module",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "floquet",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "yaml_io",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines the utils package surface so callers can import reconstruct_floquet and YAML helpers without reaching into submodules. It keeps the public API small and stable while internal PETSc and provenance helpers remain private or opt-in.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/utils/__init__.py",
          "importance_score": 0.45,
          "interfaces": [],
          "name": "__init__.py",
          "responsibilities": [
            "Expose public utils API",
            "Re-export Floquet helper",
            "Re-export YAML config helpers"
          ],
          "source_summary": "Contains a short module docstring and from-import re-exports of reconstruct_floquet plus load_config and typed list parsers from yaml_io. No local logic beyond packaging the public symbols.",
          "summary": "Package entry that re-exports the public utility API for Floquet reconstruction and YAML config parsing."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "dataclasses",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "collections.abc",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements a dataclass-backed configuration object and helpers so eigenvalue, resolvent, and linear-solve paths read settings the same way. YAML values win over environment variables, which win over project defaults, reducing drift between solver entry points.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/utils/_solver_cfg.py",
          "importance_score": 0.86,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "SolverConfig",
              "parameters": [],
              "return_type": "SolverConfig",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "from_config",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cfg",
                  "param_type": "SolverConfig | Mapping[str, Any] | None"
                }
              ],
              "return_type": "SolverConfig | None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "__getitem__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "key",
                  "param_type": "str"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "subsection",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "env_prefix",
                  "param_type": "str"
                }
              ],
              "return_type": "Mapping[str, Any] | None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "as_bool",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str | None"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            }
          ],
          "name": "_solver_cfg.py",
          "responsibilities": [
            "Normalize solver configuration inputs",
            "Enforce YAML then env then default precedence",
            "Provide typed config accessors",
            "Support mapping-style config iteration"
          ],
          "source_summary": "Defines SolverConfig with mapping-like access, subsection lookup, and typed readers such as _scfg, _scfg_float, _scfg_int, and _scfg_bool. Includes from_config conversion and as_bool parsing so callers can normalize mixed config inputs safely.",
          "summary": "Shared SolverConfig policy that unifies YAML, environment, and default precedence for PETSc/SLEPc callers."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "subprocess",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "resource",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides portable measurement helpers so solver runs can attach comparable resource usage to outputs. It handles POSIX RSS differences and multiple GPU memory backends including CuPy and nvidia-smi fallbacks.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/utils/cost_metrics.py",
          "importance_score": 0.62,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "peak_rss_bytes",
              "parameters": [],
              "return_type": "int",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_gpu_memory_bytes",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cost_gpus",
                  "param_type": "int"
                }
              ],
              "return_type": "float",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_petsc_options_string",
              "parameters": [],
              "return_type": "str",
              "visibility": ""
            }
          ],
          "name": "cost_metrics.py",
          "responsibilities": [
            "Measure peak process RSS",
            "Probe GPU memory usage",
            "Integrate with PETSc communicator context",
            "Support cost metadata on artifacts"
          ],
          "source_summary": "Implements peak_rss_bytes with resource module handling, PETSc communicator helpers, and GPU memory probes via CuPy or nvidia-smi. Also builds PETSc options strings used when recording run cost context.",
          "summary": "Collects uniform runtime cost metrics such as peak RSS and GPU memory for saved result artifacts."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Supplies the domain-specific helper used after stability computations that involve Floquet exponents. It maps active-grid mode values back onto full coordinates so downstream analysis and visualization see physical modes.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/utils/floquet.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "reconstruct_floquet",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "X",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "V",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "used_ind",
                  "param_type": "np.ndarray"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "floquet_exp",
                  "param_type": "float"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            }
          ],
          "name": "floquet.py",
          "responsibilities": [
            "Reconstruct Floquet modes",
            "Map active grid values to physical space",
            "Apply Floquet exponential scaling"
          ],
          "source_summary": "Exposes reconstruct_floquet(X, V, used_ind, floquet_exp) using NumPy arrays. Accepts one-dimensional or multi-component mode values on active points and rebuilds the physical-space representation with the Floquet exponential factor.",
          "summary": "Reconstructs Floquet modes from computational-grid values into physical-space fields."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "scipy.sparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "petsc4py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Largest and most central module in utils; it bridges SciPy sparse operators and PETSc matrices for local and distributed workflows. Shared placement avoids circular imports between mesh and solver layers while encoding complex-mode and distribution policy.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/utils/petsc_core.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "scipy_to_petsc",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mat",
                  "param_type": "sp.spmatrix"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "solver_cfg",
                  "param_type": "dict | None"
                }
              ],
              "return_type": "PETSc.Mat",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "petsc_to_scipy",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mat",
                  "param_type": "PETSc.Mat"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "format",
                  "param_type": "str"
                }
              ],
              "return_type": "sp.csr_matrix | sp.csc_matrix",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "compact_petsc_matrix",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mat",
                  "param_type": "PETSc.Mat"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "solver_cfg",
                  "param_type": "dict | None"
                }
              ],
              "return_type": "PETSc.Mat",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "petsc_vec",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "arr",
                  "param_type": "np.ndarray"
                }
              ],
              "return_type": "PETSc.Vec",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "vec_to_numpy",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "v",
                  "param_type": "PETSc.Vec"
                }
              ],
              "return_type": "np.ndarray",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "require_complex",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "solver_cfg",
                  "param_type": "dict | None"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "assemble_matrix",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "matrix",
                  "param_type": "PETSc.Mat"
                }
              ],
              "return_type": "PETSc.Mat",
              "visibility": ""
            }
          ],
          "name": "petsc_core.py",
          "responsibilities": [
            "Convert SciPy sparse matrices to PETSc",
            "Convert PETSc matrices back to SciPy",
            "Manage distributed ownership and assembly",
            "Expose vector and matrix introspection helpers",
            "Apply solver-backend matrix mode policy"
          ],
          "source_summary": "Provides lazy PETSc access, scipy_to_petsc and petsc_to_scipy conversions, matrix compaction, ownership ranges, vector helpers, nnz and shape introspection, and MUMPS ICNTL option mapping. Includes require_complex and distributed-matrix mode decisions driven by solver_cfg.",
          "summary": "Core SciPy to PETSc conversion and matrix/vector utilities shared by mesh filters and solvers."
        },
        {
          "code_purpose": "middleware",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "petsc4py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "slepc4py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Keeps top-level package import PETSc-free while ensuring solver modules can initialize backends safely when a PETSc path runs. Handles already-initialized cases and sanitizes argv before parsing.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/utils/petsc_init.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "init_petsc_from_argv",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "argv",
                  "param_type": "list[str] | None"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "init_slepc_from_argv",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "argv",
                  "param_type": "list[str] | None"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_sanitize_petsc_argv",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "argv",
                  "param_type": "list[str]"
                }
              ],
              "return_type": "list[str]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_already_initialized",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "exc",
                  "param_type": "Exception"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            }
          ],
          "name": "petsc_init.py",
          "responsibilities": [
            "Initialize PETSc once per process",
            "Initialize SLEPc once per process",
            "Sanitize argv for backend parsers",
            "Avoid import-time PETSc coupling"
          ],
          "source_summary": "Tracks _PETSC_INIT_DONE and _SLEPC_INIT_DONE flags and exposes init_petsc_from_argv and init_slepc_from_argv. Helper functions detect prior initialization errors and clean argv lists before backend init.",
          "summary": "Initializes petsc4py and slepc4py exactly once from process argv so solver CLI options reach the stack."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "hashlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "re",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "subprocess",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "importlib.metadata",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Gathers git SHA, config hash, host, compute mode, MPI stack, package versions, PETSc arch, MUMPS, and BLAS details into a stable key set. Result writers can embed this dictionary so experiments remain auditable across environments.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/utils/provenance.py",
          "importance_score": 0.72,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "collect_provenance",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "config_path",
                  "param_type": "str | Path | None"
                }
              ],
              "return_type": "dict[str, str]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_git_head_sha",
              "parameters": [],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_config_md5",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "config_path",
                  "param_type": "str | Path | None"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_package_version",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "distribution",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            }
          ],
          "name": "provenance.py",
          "responsibilities": [
            "Capture git and config fingerprints",
            "Probe numerical stack versions",
            "Record host and compute mode",
            "Assemble provenance dictionaries for artifacts"
          ],
          "source_summary": "Defines _PROVENANCE_KEYS and probe helpers for git head, config MD5, package versions via importlib.metadata, PETSc arch, MUMPS, MPI implementation, BLAS linkage, and hostname. collect_provenance aggregates these into a string-valued dictionary.",
          "summary": "Collects runtime provenance metadata for persisted result artifacts to support reproducibility."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "re",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "yaml",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Front door for case configuration input, including a SafeLoader extension that accepts unsigned scientific floats common in scientific YAML. Parsing helpers convert loosely typed YAML values into Python complex and int lists expected by solvers.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/utils/yaml_io.py",
          "importance_score": 0.84,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "load_config",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "str | Path"
                }
              ],
              "return_type": "dict",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "parse_complex",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                }
              ],
              "return_type": "complex",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "parse_complex_list",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                }
              ],
              "return_type": "list[complex]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "parse_int_list",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                }
              ],
              "return_type": "list[int]",
              "visibility": ""
            }
          ],
          "name": "yaml_io.py",
          "responsibilities": [
            "Load YAML case configuration files",
            "Accept scientific float variants",
            "Parse complex scalars and lists",
            "Parse integer lists from YAML values"
          ],
          "source_summary": "Defines _CaseLoader with an implicit resolver for scientific floats, load_config(path) returning a dict, and parse_complex, parse_complex_list, and parse_int_list utilities. Uses PyYAML plus pathlib and regex for robust case-file ingestion.",
          "summary": "Loads YAML case files and parses complex numbers and integer or complex lists used in configs."
        }
      ],
      "importance_score": 0.88,
      "key_files": [
        "petsc_core.py",
        "_solver_cfg.py",
        "yaml_io.py",
        "petsc_init.py",
        "provenance.py"
      ],
      "name": "utils",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy/utils",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The utils package is shared backend infrastructure for linstabpy linear-stability workflows. It centralizes YAML case I/O, solver config precedence, PETSc/SLEPc initialization and matrix conversion, Floquet reconstruction, runtime cost metrics, and provenance capture. These modules are imported across mesh and solver layers so configuration, numerical backends, and result metadata stay consistent."
    },
    {
      "file_count": 0,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [],
          "detailed_description": "PKG-INFO is the primary distribution metadata file for the linstabpy egg/wheel packaging layout. It identifies the project as stability and resolvent analysis software for compressible Navier-Stokes flows and points to repository, docs, and issue trackers. It is consumed by installers and indexes rather than executed as application code.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy.egg-info/PKG-INFO",
          "importance_score": 0.4,
          "interfaces": [],
          "name": "PKG-INFO",
          "responsibilities": [
            "Declare package identity and version",
            "Record license and project URLs",
            "Expose keywords and classifiers for discovery",
            "Document high-level package purpose"
          ],
          "source_summary": "Contains Metadata-Version 2.4 fields: Name linstabpy, Version 1.0.0rc9, Summary about compressible viscous flow stability, Authors Ricardo Frantz and Gabriel Hamada, License-Expression GPL-3.0-only, Project-URLs, Keywords (stability, resolvent, PETSc, SLEPc), and classifiers. Content is declarative key-value metadata, not executable Python.",
          "summary": "Core package metadata describing name, version, summary, authors, license, URLs, keywords, and classifiers for linstabpy."
        },
        {
          "code_purpose": "config",
          "dependencies": [],
          "detailed_description": "SOURCES.txt enumerates project files packaged with linstabpy, from root docs and pyproject.toml through src/linstabpy modules and this egg-info tree. It helps reconstruct package layout and confirms which modules (e.g., baseflow) ship with the release. It is build/install bookkeeping, not runtime logic.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy.egg-info/SOURCES.txt",
          "importance_score": 0.32,
          "interfaces": [],
          "name": "SOURCES.txt",
          "responsibilities": [
            "List packaged source paths",
            "Record inclusion of docs and license files",
            "Mirror package layout for sdist/install",
            "Include egg-info self-references"
          ],
          "source_summary": "Plain-text path list including CITATION.cff, LICENSE, README.md, pyproject.toml, src/linstabpy package modules (__init__.py, metrics_schema.py, mumps_options.py, baseflow subpackage files), and the egg-info metadata files themselves. Entries are truncated in the provided excerpt but follow standard setuptools SOURCES format.",
          "summary": "Manifest listing every source and metadata path included in the package distribution."
        },
        {
          "code_purpose": "config",
          "dependencies": [],
          "detailed_description": "dependency_links.txt exists for historical setuptools support of non-PyPI dependency URLs. In this package it is empty, indicating no legacy dependency links are configured. It has negligible runtime or architectural impact.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy.egg-info/dependency_links.txt",
          "importance_score": 0.1,
          "interfaces": [],
          "name": "dependency_links.txt",
          "responsibilities": [
            "Reserve legacy dependency-link slot",
            "Indicate absence of custom dependency URLs"
          ],
          "source_summary": "File content is blank/whitespace only. No URLs or link entries are declared.",
          "summary": "Placeholder file for legacy setuptools dependency links; currently empty."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "numpy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "scipy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "matplotlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "pyyaml",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "h5py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "mpi4py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "petsc4py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "slepc4py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "nvmath-python",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "requires.txt specifies core scientific stack requirements and optional feature extras used by installers. Core deps cover numerics and config (numpy, scipy, matplotlib, pyyaml); extras gate GPU (cudss), HDF5 I/O, and parallel PETSc/SLEPc stacks. This file is central to understanding the package's external integration surface.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy.egg-info/requires.txt",
          "importance_score": 0.38,
          "interfaces": [],
          "name": "requires.txt",
          "responsibilities": [
            "Define mandatory install dependencies",
            "Declare optional extras for PETSc, HDF5, and cuDSS",
            "Constrain dependency version ranges",
            "Guide environment setup for scientific backends"
          ],
          "source_summary": "Lists numpy>=1.23, scipy>=1.10, matplotlib>=3.7, pyyaml>=6.0 as base requirements. Optional sections: [cudss] nvmath-python[cu12], [hdf5] h5py, [petsc] mpi4py/petsc4py/slepc4py with version lower bounds. Format is setuptools requires.txt with extras.",
          "summary": "Declares runtime and optional extra dependencies for installing linstabpy."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "linstabpy",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "top_level.txt tells packaging tools which top-level import name the wheel or egg provides. Here it records a single package, linstabpy, aligning install metadata with the src layout. It is a small but standard packaging marker.",
          "file_path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy.egg-info/top_level.txt",
          "importance_score": 0.25,
          "interfaces": [],
          "name": "top_level.txt",
          "responsibilities": [
            "Declare top-level import package name",
            "Align distribution metadata with src package",
            "Support installer namespace discovery"
          ],
          "source_summary": "Single line containing the identifier linstabpy. No additional top-level modules are listed.",
          "summary": "Names the top-level Python package export exposed by the distribution."
        }
      ],
      "importance_score": 0.28,
      "key_files": [
        "PKG-INFO",
        "requires.txt",
        "SOURCES.txt",
        "top_level.txt",
        "dependency_links.txt"
      ],
      "name": "linstabpy.egg-info",
      "path": "/home/rfrantz/openfluids/linstabpy/src/linstabpy.egg-info",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The linstabpy.egg-info directory holds setuptools-generated packaging metadata for the linstabpy scientific Python package (stability and resolvent analysis of compressible viscous flows). Its files declare package identity, version, license, dependency requirements (including optional PETSc/SLEPc and HDF5 extras), the top-level import name, and the full source file manifest used at build/install time. Together they describe how the package is distributed and what it depends on, without containing runtime application logic."
    }
  ],
  "file_insights": []
}
```

## Memory Storage Statistics

**Total Storage Size**: 762292 bytes

- **preprocess**: 243645 bytes (32.0%)
- **documentation**: 353203 bytes (46.3%)
- **studies_research**: 165407 bytes (21.7%)
- **timing**: 37 bytes (0.0%)

## Generated Documents Statistics

Number of Generated Documents: 14

- Architecture Description
- Key Modules and Components Research Report_Case Orchestration
- Key Modules and Components Research Report_Linear Algebra Infrastructure
- Project Overview
- Key Modules and Components Research Report_Baseflow Management
- Key Modules and Components Research Report_Stability Analysis
- Key Modules and Components Research Report_Configuration Management
- Key Modules and Components Research Report_Mesh Discretization
- Core Workflows
- Key Modules and Components Research Report_Resolvent Analysis
- Key Modules and Components Research Report_LNSE Operator Assembly
- Key Modules and Components Research Report_Solver Routing and Sweep Optimization
- Boundary Interfaces
- Key Modules and Components Research Report_Shared Scientific Utilities
