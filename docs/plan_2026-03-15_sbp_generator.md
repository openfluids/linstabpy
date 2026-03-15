# Plan: SBP Automatic Coefficient Generator

Date: 2026-03-15

## Goal

Replace hardcoded SBP stencil lookup tables with algorithmic generation so any
fd_order works without manually transcribing coefficients.

## Phase 2a — Fornberg interior stencils

1. Implement Fornberg (1988) algorithm for FD weights from Taylor series
2. Generate `a_i1` (1st derivative) and `a_i2` (2nd derivative) for any even order
3. Validate against existing hardcoded tables for orders 2, 4, 6, 8

## Phase 2b — SBP boundary closure solver

1. Build the diagonal-norm H matrix for the SBP boundary region
2. Set up the constrained system: Taylor accuracy (order p/2 at boundary) + SBP property
3. Solve the underdetermined system (Mattsson & Nordström 2004 approach)
4. Validate against existing tables for orders 2, 4, 6, 8 to machine precision

## Files

- Create: `src/linstab2d/mesh/sbp_generator.py`
- Modify: `src/linstab2d/mesh/sbp.py` (delegate to generator)
- Create: `tests/test_sbp_generator.py`

## Verification

- Generated coefficients for orders 2–8 must match hardcoded tables to machine precision
- D1/D2 on polynomials of degree p must be exact
- SBP property: `Q + Q^T = B` for each generated operator
- Parametric sweep: orders 2 through 16
