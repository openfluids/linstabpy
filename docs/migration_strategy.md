# Migration Strategy for Architecture Refactor

This strategy controls the transition from the current assembly/runtime helpers
to the BCRecipe, LNSESystem, SolverConfig, runtime-pipeline, and Mesh-state
seams. It is written for maintainers and contributors working on the refactor,
not as a replacement for the public README.

## Compatibility Window

Existing public entry points remain supported for at least one full minor
release after the new seams land:

- `build_lnse_operator`
- `build_lnse_operator_petsc`
- `build_lnse_system`
- documented high-level runtime/example JSONC workflows
- existing list-of-dict boundary-condition recipes loaded from JSONC

During that window, the old names become compatibility wrappers. They may
delegate internally to new typed objects, but their accepted inputs, returned
values, and numerical results must remain unchanged unless a bug fix is
explicitly documented.

No old public entry point is deleted until:

1. the Architecture Refactor Test Matrix passes for the affected phase;
2. the LNSE equivalence harness passes on the current baseline cases;
3. the dual-path comparison for the migrated entry point passes;
4. the release notes and README name the replacement API and compatibility
   window.

## Shim Policy

Use narrow wrappers rather than parallel implementations:

- `build_lnse_operator(...)` should remain the lightest wrapper for callers that
  need only the assembled operator and index map.
- `build_lnse_operator_petsc(...)` should preserve PETSc-specific behavior,
  matrix ownership, and MPI assumptions.
- `build_lnse_system(...)` is the natural bridge to the richer internal system
  object because it already returns the operator plus boundary-condition data.
- JSONC boundary-condition recipe lists remain valid input. A future BCRecipe
  object may be constructed internally from those dicts, but users should not
  need to change existing configs during the compatibility window.

Wrappers should be thin enough that the new seam owns policy. If a wrapper
starts duplicating boundary-condition ordering, backend selection, PETSc/SLEPc
configuration, or NPZ provenance logic, the policy belongs in the seam instead.

## Dual-Path Gate

When an entry point is migrated, run old and new paths on the same inputs and
compare:

- coefficient tensors when the change touches coefficient preparation;
- raw assembled LNSE blocks before `inv(B)`, sponge, dissipation, and BCs;
- final pre-BC and post-BC operators;
- BC effective and column-zeroed index sets;
- downstream spectra, gains, residuals, accepted/rejected mode indices, and
  selected NPZ provenance fields when the runtime path is affected.

The first checked-in gate is:

```bash
.venv/bin/python scripts/validation/lnse_equivalence_harness.py \
    --case 2D --case axy \
    --output artifacts/architecture_refactor/safety_infrastructure/lnse_equivalence_harness.json
```

The combined safety gate is:

```bash
.venv/bin/python scripts/validation/architecture_refactor_matrix.py \
    --group safety-infrastructure \
    --report artifacts/architecture_refactor/safety_infrastructure/matrix_safety_infrastructure.json
```

Before candidate paths exist, the harness compares the current path against
itself. After a seam adds a candidate path, the same report schema must compare
old versus new. A tolerance change is not a normal migration tool; record the
baseline, candidate, and numerical reason in the bead before accepting it.

## Deprecation Policy

Deprecation warnings are allowed only after replacement APIs are documented and
the compatibility wrappers have passing dual-path evidence. Warnings should be:

- specific about the replacement;
- tied to a release window;
- absent from hot inner loops;
- easy to silence by updating the call site rather than by setting global
  filters.

Internal callers can move to the new seams earlier than external users, but only
after their migrated path has a passing matrix slice. Public examples should
stay on stable documented APIs until the new seam is ready for normal users.

## Bead Review Note

For each migrated entry point, record:

- git SHA;
- old entry point and new seam used;
- matrix slice and dual-path command;
- operator/BC/eigen/resolvent/NPZ diff summary;
- documentation gate result;
- whether the public call remains unchanged, warns, or is intentionally
  deferred.
