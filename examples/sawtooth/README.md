# Lui 2022 suction-side SWBLI (sawtooth)

Mean flow from Lui, Ricciardi, Wolf, Braun, Rahbari and Paniagua,
Phys. Rev. Fluids 7, 094602 (2022). The public field is a symmetric
stride-2 coarsening of the archived 599 x 679 mesh: 300 x 340 nodes,
already packed in `sawtooth.npz`. The driver does not apply a further
stride.

```
.venv/bin/python examples/sawtooth/sawtooth_fig1_baseflow.py
mpiexec -n 4 python -m examples.sawtooth.sawtooth --config examples/sawtooth/sawtooth_evp.yaml
.venv/bin/python examples/sawtooth/sawtooth_fig2_spectrum.py
```

The Floquet shift default is LinStab2D (`matlab_parity`). Continuum form:
`numerics.floquet_shift: continuum` or `LINSTABPY_FLOQUET_SHIFT=continuum`.
