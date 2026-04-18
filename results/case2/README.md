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
- **`tutorial_ducros.md`** — variante com sensor Ducros (histórico; ver
  `REFERENCE.md` Framing §3 sobre por que Ducros não é cleanup de
  mean-flow).
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

- `parity_config.jsonc` — configuração canônica do caso
- `case2.py` — entrada de um caso só
- `perf_memory.py` — orquestrador de campanha
- `run_case_once.sh`, `run_perf_queue_once.sh` — wrappers
- `evp_stride*.jsonc` — configs por experimento (malha × target × solver)
- `plot_saved_spectra.py`, `plot_saved_modes.py` — pós-processamento

Configs retirados do caminho ativo vivem em `legacy/`.

## Commands

Rodar a campanha ativa:

```bash
LINSTAB2D_CASE2_PERF_CONFIG=results/case2/evp_stride2_spectra.jsonc \
./.venv/bin/python results/case2/perf_memory.py
```

Plotar o espectro salvo da campanha ativa:

```bash
./.venv/bin/python results/case2/plot_saved_spectra.py \
  --run-dir results/case2/output/evp_stride2_spectra/runs
```

Plotar os modos salvos da campanha ativa:

```bash
./.venv/bin/python results/case2/plot_saved_modes.py \
  --run-dir results/case2/output/evp_stride2_spectra/runs
```

## Delete Policy

O topo de `results/case2` deve continuar pequeno. Se um arquivo não pertence à
campanha ativa em `stride 2`, ele deve ir para `results/case2/legacy/`.
