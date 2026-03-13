# Plan: Song 2023 Stability Recovery

Date: 2026-03-13

## Current assessment

The `validation/song_2023` case is now executable end-to-end for stability, but the current direct modes are too noisy to treat as validated. The main issue is therefore no longer runtime or eigensolver availability; it is result credibility.

Current Python capability:

- load baseflow from saved arrays,
- build the deformed mesh,
- assemble the LNSE operator,
- apply BCs,
- build the Chu norm,
- run stability,
- optionally run resolvent,
- save NPZ and timing logs.

Relevant implementation:

- `validation/song_2023/song_2023.py`
- `validation/song_2023/baseflow_loader.py`
- `src/linstab2d/solvers/eigenvalue.py`

## Main conclusion

The current Python branch should be treated as a **raw baseline**, not yet as a method-faithful Song reproduction.

Why:

1. the full stability solve completes,
2. but the resulting direct modes look numerically contaminated,
3. the current Python path still loads the baseflow too directly,
4. and the bundled Song helper workflow includes conditioning hooks that are not yet reproduced in Python.

Most likely missing ingredients:

- shock-related treatment,
- sponge handling/scaling,
- repeated explicit filtering of baseflow variables.

## Method gap

The bundled Song-side assets indicate a broader workflow than the current Python branch reproduces, including:

- Strouhal-to-omega conversion with characteristic ratios,
- optional sponge logic,
- optional shock-related controls,
- optional filtering passes during preprocessing.

Relevant references:

- `validation/song_2023/matlab_reference/main.m`
- `validation/song_2023/matlab_code/macro_single.sh`
- `validation/song_2023/matlab_code/BaseFlow/CreateBaseFlow/make_baseflow.py`

So the working hypothesis is: the noisy modes are more likely caused by an under-conditioned baseflow/operator input than by a fundamentally broken eigensolver.

## Constraints

- Stability is the priority; resolvent is secondary for now.
- The raw branch should be preserved as a baseline.
- We should improve conditioning before attempting more invasive solver-side filtering.
- Future stability and resolvent should still share one coherent operator path.

## Plan

### Phase 0 — documentation traceability

Goal: distinguish paper requirements from helper-script behavior.

Tasks:

1. Create a searchable local Song text companion if possible.
2. Record which numerical choices come from the paper and which come only from helper scripts.
3. Record explicitly that resolvent is our extension, not a reported Song result.

### Phase 1 — freeze the raw baseline

Goal: preserve the current branch for comparison.

Tasks:

1. Keep the current `song_2023.jsonc` setup identifiable as the raw baseline.
2. Retain the current full stability-only artifact/log as the baseline reference.
3. Compare future runs against this branch instead of silently replacing it.

### Phase 2 — add Python preprocessing

Goal: reproduce the missing conditioning layer before `BaseFlow` construction.

Tasks:

1. Extend `validation/song_2023/baseflow_loader.py` with a config-driven preprocessing step.
2. Add a `baseflow_preprocess` block to `validation/song_2023/song_2023.jsonc`.
3. Support, in this order:
   - sponge scaling/regeneration,
   - repeated explicit filtering,
   - optional shock mask/damping.
4. Log exactly which preprocessing options were applied.

### Phase 3 — validate stability first

Goal: determine whether preprocessing improves the direct modes.

Tasks:

1. Run a reduced experiment ladder:
   - raw,
   - filter-only,
   - sponge-only,
   - filter + sponge,
   - shock treatment if still needed.
2. Compare eigenvalue movement and direct-mode smoothness.
3. Add quantitative diagnostics for mode roughness/localization instead of relying only on plots.

### Phase 4 — return to full resolution

Goal: rerun the full stability case only after the reduced tests identify a credible conditioned branch.

Tasks:

1. Promote the best reduced preprocessing setup.
2. Run full-resolution stability.
3. Compare against MATLAB and scaled Song references.
4. Accept the branch only if the modes are both cleaner and spectrally defensible.

### Phase 5 — re-enable resolvent

Goal: add resolvent only after the stability branch is credible.

Tasks:

1. Reuse the conditioned operator/baseflow path.
2. Run resolvent as an explicit extension of the stabilized case.
3. Keep paper-vs-extension labeling clear in outputs and notes.

## Immediate next actions

1. Preserve the current raw stability artifact as the baseline.
2. Add `baseflow_preprocess` to the Song JSONC schema.
3. Implement filter-only preprocessing first.
4. Run reduced stability diagnostics before touching resolvent again.

## Success criterion

The Song case is only considered recovered when:

- the full stability run completes,
- the direct modes are no longer visibly dominated by numerical roughness,
- the spectrum is defensible against the available references,
- and the preprocessing choices are explicit and reproducible.
