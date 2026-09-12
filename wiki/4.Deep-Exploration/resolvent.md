# Resolvent

| File | Role |
|---|---|
| `resolvent.analysis` | `resolvent_analysis`, leading gain at one frequency |
| `resolvent.operator` | Resolvent linear operator |
| `resolvent.gain` | Gain from forcing/response |
| `resolvent.conservative_forcing` | Conservative-variable forcing option |
| `solvers.router.resolvent_map` | Leading-gain map over \(\beta\) and frequency |

`resolvent_map` walks each spanwise-wavenumber line. It is not the 1-D
`amortized_stable_sweep` tool. Mesh must be supplied as `cfg['mesh']`,
`cfg['mesh_config']`, or `base_flow.mesh`.

Chu weights `W_diag` define the energy inner product for the gain.
