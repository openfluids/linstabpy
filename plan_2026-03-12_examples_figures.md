# Plan 2026-03-12: Remaining Verification for Example Figures

Shared plotting helpers and the main round of case-specific layout/data fixes are already in place. This file now tracks only the remaining verification work.

## Remaining tasks

1. Regenerate the touched example outputs from the current scripts and caches.
2. Run `./run_all_examples.sh` as the canonical acceptance check.
3. Re-run the parity comparison step after the example pass completes.
4. Review the regenerated PNGs against `examples/fix_examples_figures.md` item by item.
5. Record any still-deferred comments explicitly instead of leaving ambiguous outputs in the tree.

## Guardrails

1. Do not delete cached `*.npz` or older figure artifacts until a verified replacement exists.
2. Keep figure-verification work separate from unrelated case-study or paper edits.
3. If a remaining issue is really a physics/data problem rather than a layout problem, leave it explicit instead of masking it with plotting changes.

## Acceptance

1. All non-deferred review comments are either visibly fixed or explicitly deferred.
2. The canonical example runner still completes.
3. Parity comparison still runs on the regenerated outputs.
