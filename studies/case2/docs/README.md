# Case 2 EVP Campaign

## Comece Aqui

Este README é **operacional** — como rodar. Para o resto:

- **`PLAN.md`** (inglês) — estado atual da campanha, critérios de aceitação,
  plano por fases com status, TODOs abertos, lista de "não faça".
- **`REFERENCE.md`** (inglês) — pano de fundo estável: linhagem do código
  (porte do Martini & Schmidt 2024), equações, normas, *Physical and
  Numerical Framing* (resolvent, mean-flow vs base-flow, gap LES/DNS),
  parameter map, referências.
- **`tutorial.md`** — passo-a-passo prático (rodar um caso, rodar uma
  campanha, MPI sem pegadinha).
- **`/CLAUDE.md`** — motto e regras globais do projeto.

Se você é um agente entrando nesta campanha agora, leia `CLAUDE.md`,
`PLAN.md`, depois `REFERENCE.md`. Volte aqui para comandos.

## Artifact Layout

Cada campanha grava em `output/<campanha>/`:

- `runs/` — um `.npz` e um `.config.json` por run
- `logs/` — um log por run
- `plots/spectra/`, `plots/modes/` — figuras
- arquivos-resumo no topo (JSON latest, CSV table)

## Arquivos ativos nesta pasta

- `case2.py` — entrada de um caso só
- `perf_memory.py` — orquestrador de campanha
- `run_case_once.sh`, `run_perf_queue_once.sh` — wrappers
- `configs/parity_config.jsonc` — configuração canônica do caso
- `configs/evp/evp_stride*.jsonc` — configs por experimento (malha × target × solver)
- `scripts/plot_saved_spectra.py`, `scripts/plot_saved_modes.py` — pós-processamento

## Commands

Rodar a campanha ativa (exemplo: M21 com target oscilatório):

```bash
LINSTAB2D_CASE2_PERF_CONFIG=studies/case2/configs/evp/evp_stride2x1_oscillatory_target_ncv500_tol1e4.jsonc \
./.venv/bin/python studies/case2/perf_memory.py
```

Plotar o espectro salvo:

```bash
./.venv/bin/python studies/case2/scripts/plot_saved_spectra.py \
  --run-dir studies/case2/output/evp_stride2x1_oscillatory_target_ncv500_tol1e4/runs
```

Plotar os modos salvos:

```bash
./.venv/bin/python studies/case2/scripts/plot_saved_modes.py \
  --run-dir studies/case2/output/evp_stride2x1_oscillatory_target_ncv500_tol1e4/runs
```

## Delete Policy

O topo de `studies/case2` deve continuar pequeno. Configs ou saídas que
saírem do caminho ativo devem ser apagados, não movidos para um `legacy/`
permanente. Histórico fica no git.
