# Run logs behind the paper tables

Raw Slurm logs, job scripts, and configs of the runs behind the numbers in the
LinStabPy paper, recomputed on the release tree (`06256e4`, 1.0.0rc9+15) on
2026-09-14 and 15, plus a full rerun of the Savarino campaigns on `v1.0.0rc10`
on 2026-09-16. One directory per job, `<date>_<job id>_<what>/`, holding
`slurm-<id>.out`, `slurm-<id>.err`, and the `.sbatch` that ran it. `common.sh`
is sourced by every job: oneAPI environment, one thread per rank, Intel MPI
pinning off with the pin map printed, and the release tree placed ahead of the
virtual environment on `PYTHONPATH`.

All jobs ran on one node, c9: dual Xeon Silver 4210R (20 cores, 40 hardware
threads), two NVIDIA RTX A6000, Slurm partition `LocalQ`. The 2026-09-14 Slurm
logs call it "workstation-B"; the run-record CSVs call it `c9`. It is the same
machine. In the logs, `<repo>` is this repository and `<venv>` its virtual
environment.

| Directory | Job | What it produces | Paper |
|---|---|---|---|
| `2026-09-14_3073_martini_table1` | `run_all_examples.sh`, serial | the six Martini examples and `examples/martini/martini_parity_table.csv` | Table 1 |
| `2026-09-14_3074_example5_ordering` | Example 5 under MUMPS orderings default, metis, pord, amd, qamd | gain sensitivity to the fill-reducing ordering | Example 5 discussion |
| `2026-09-14_3075_song_fd4_n30` | `examples/song/configs/run_fd4_n30.yaml`, 4 ranks, then the same field scaled by the speed of sound as a control (`run_fd4_n30_cinf.yaml`) | the 32 degree leading eigenvalue against Song (2023) | Section 4 |
| `2026-09-14_3076_mack_f2_b45_n4` | `examples/savarino/configs/mack_f2_b45.yaml`, 4 ranks | gain, wall time, peak memory (sum and per-rank maximum) | Appendix B |
| `2026-09-14_3077_mack_f2_b45_n8` | same, 8 ranks | same | Appendix B |
| `2026-09-14_3078_mack_f2_b45_n16` | same, 16 ranks | same | Appendix B |
| `2026-09-14_3082_mack_f2_b45_gpu` | `examples/savarino/savarino_point_cudss.yaml`, one host rank, ten threads, one RTX A6000, two repetitions | the same point on the GPU: wall time, host and device memory, gain | Appendix B |
| `2026-09-14_3083_cpu_line_b40_n4` | `savarino_mapwalk.py --n-anchor 99 --ncv 5`, β = 40 line, 4 ranks, MUMPS | line-mode cost per frequency on the CPU | Appendix B figure |
| `2026-09-14_3084_cpu_line_b40_n8` | same, 8 ranks | same | Appendix B figure |
| `2026-09-14_3085_cpu_line_b40_n16` | same, 16 ranks | same | Appendix B figure |
| `2026-09-14_3079_cudss_line_b40` | same line on one RTX A6000 with cuDSS, default refactorization path | line-mode cost per frequency on the GPU | Appendix B figure |
| `2026-09-14_3081_cudss_line_b40_fast` | same with the streamlined refactorization (`LINSTABPY_FAST_REFACTOR=1`, an opt-in patch at the time and the default since 1.0.0rc10) | the open diamond of the figure | Appendix B figure |
| `2026-09-15_sub47` | the 47-point subset (β = 40 and 45 lines) as four arms one after another: CPU independent jobs (4 ranks, 5 concurrent), CPU lines (2 × 4 ranks), cuDSS independent jobs (2 concurrent), cuDSS lines (streamlined refactorization); collected run records, the arm and cascade scripts, and the cascade log | a rerun of the 47-point rows of the resolvent cost table on the release tree | none: Table 2 quotes the August campaign; this rerun gives 144.2 s (CPU) and 56.5 s (cuDSS) per independent point |
| `2026-09-15_map200` | the full 200-point map as four arms one after another: CPU independent jobs (4 ranks, 5 concurrent), CPU lines (14 lines, 2 concurrent), cuDSS independent jobs (2 concurrent), cuDSS lines (streamlined refactorization, 2 concurrent); collected run records, the arm and cascade scripts, the cascade logs | a rerun of the 200-point rows of the resolvent cost table on the release tree | none: Table 2 quotes the August campaign; this rerun gives 138.6 s (CPU) and 57.6 s (cuDSS) per independent point |
| `2026-09-16_rc10` | every Savarino campaign again on the tagged `v1.0.0rc10`: the 200-point map (CPU and cuDSS, independent jobs and lines), the 317-point CPU map, the 47-point subset (four arms), and the β = 40 line at 4, 8 and 16 ranks and on one A6000; collected run records, arm and cascade scripts, cascade log | a rerun of the Section 6 cost campaign on a tagged release | none: Table 2 quotes the August campaign (`examples/savarino/savarino_campaign_walltimes.csv`) |

The jobs ran one at a time on an otherwise idle node (`afterany` chain), so
every wall time in a log is a single-tenant measurement. The scaling jobs
print `peak_rss_sum_bytes` and `peak_rss_max_bytes` from a three-line addition
to `driver/run.py` (an `MPI.MAX` allreduce next to the existing `MPI.SUM`).
