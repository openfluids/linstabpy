# Tutorial de Shock Cleanup (sensor Ducros) — Case 2

Tutorial **separado** do `tutorial.md` porque o `shock_cleanup` não está no
caminho ativo do Case 2. É um experimento controlado: aplica um sensor
Ducros sobre o baseflow, opcionalmente filtra a região de choque, e roda
análise de estabilidade/resolvent **duas vezes** (baseflow raw vs. cleaned)
lado a lado.

Este tutorial assume que você quer testar o cleanup **em cima do baseflow já
existente do Case 2**, sem montar um caso novo do zero.

Use este guia quando quiser:

- entender o que o sensor Ducros faz no baseflow
- calibrar limiares (`ducros_threshold`, `compression_threshold`)
  visualizando a máscara
- comparar um EVP/resolvent **com** e **sem** cleanup
- decidir se o cleanup remove ruído numérico ou altera física

Não use isto como default do Case 2 — veja [README.md](README.md) Phases 1-5.

## 1. O que está neste diretório

```
results/case2/ducros/
├── 01_sensor_only.jsonc     Etapa 1: só sensor, gera cleanup_mask.png pra calibrar
├── 02_sensor_filter.jsonc   Etapa 2: sensor + filtro leve, compara espectros
├── run.sh                   Wrapper — chama o runner com MPI preset e heartbeat
├── output/                  Gerado ao rodar (ignorado pelo git)
└── (este arquivo)
```

O runner de verdade é
[`../legacy/run_shock_cleanup_experiment.py`](legacy/run_shock_cleanup_experiment.py).
Ele vive em `legacy/` por política (fora do caminho ativo), mas o `run.sh`
chama ele transparentemente.

## 2. Fluxo mínimo para o Gabriel

Se a ideia é só testar o mecanismo em cima do Case 2 atual, faça exatamente
isso:

1. Entre na raiz do repositório `linstabpy/`.
2. Rode a etapa 1 em serial:

```bash
bash results/case2/ducros/run.sh results/case2/ducros/01_sensor_only.jsonc
```

3. Abra:

```text
results/case2/ducros/output/01_sensor_only/cleanup_mask.png
results/case2/ducros/output/01_sensor_only/baseflow_compare.png
results/case2/ducros/output/01_sensor_only/summary.json
```

4. Ajuste `ducros_threshold` e `compression_threshold` em
   [01_sensor_only.jsonc](ducros/01_sensor_only.jsonc) até a máscara ficar
   razoável.
5. Copie os thresholds aprovados para
   [02_sensor_filter.jsonc](ducros/02_sensor_filter.jsonc).
6. Rode a etapa 2:

```bash
bash results/case2/ducros/run.sh results/case2/ducros/02_sensor_filter.jsonc
```

7. Abra:

```text
results/case2/ducros/output/02_sensor_filter/baseflow_compare.png
results/case2/ducros/output/02_sensor_filter/stability_compare.png
results/case2/ducros/output/02_sensor_filter/resolvent_compare.png
results/case2/ducros/output/02_sensor_filter/summary.json
```

Se isso funcionar em serial, só depois vale testar MPI.

## 3. O que o sensor Ducros faz

Em três frases:

1. Calcula `div² / (div² + vort² + eps)` a partir dos gradientes de
   velocidade do baseflow — perto de 1 em compressão forte (choques), perto
   de 0 em regiões dominadas por vorticidade.
2. Combina com um limiar opcional de compressão (`div <= compression_threshold`)
   para montar a **máscara de choque**, dilatada por `buffer_cells`.
3. Aplica opcionalmente um **filtro binomial** (suaviza fora do choque) e/ou
   um **sponge shock** (dissipação extra na região marcada).

Implementação: [`../../src/linstab2d/baseflow/shock_cleanup.py`](../../src/linstab2d/baseflow/shock_cleanup.py),
função `apply_optional_shock_cleanup` em `:311`.

## 4. Fluxo recomendado em duas etapas

### Etapa 1 — calibrar o sensor visualmente

Rode com sensor ligado, filtro/sponge desligados:

```bash
bash results/case2/ducros/run.sh results/case2/ducros/01_sensor_only.jsonc
```

Outputs caem em `results/case2/ducros/output/01_sensor_only/`.

**Abra primeiro `cleanup_mask.png`.** Painéis: sensor bruto, máscara
dilatada, peso de preservação. Critérios:

- Máscara **cobre** o choque? Se não, baixe `sensor.ducros_threshold` (0.85–0.90)
  e/ou ative `sensor.compression_threshold` (por exemplo `-3.0` ou `-5.0`;
  negativo porque `div < 0` em compressão).
- Máscara **cobre demais** (camada limite, cisalhamento)? Suba
  `ducros_threshold` para 0.97-0.99.
- `summary.json` → `cleanup.sensor.mask_fraction` deve ficar em **0.5% a 3%**
  do domínio.
  Acima de 10% está frouxa demais.

Nessa etapa o runner **não roda EVP nem resolvent**. A ideia é calibrar a
máscara visualmente com custo mínimo.

Edite `01_sensor_only.jsonc` e re-rode até a máscara ficar boa.

### Etapa 2 — ligar o filtro e comparar espectros

Copie os thresholds calibrados na etapa 1 para `02_sensor_filter.jsonc` (nos
mesmos campos `sensor.*`) e rode:

```bash
bash results/case2/ducros/run.sh results/case2/ducros/02_sensor_filter.jsonc
```

Outputs em `results/case2/ducros/output/02_sensor_filter/`.

**Olhe em ordem:**

1. `baseflow_compare.png` — três colunas (raw \| cleaned \| delta) por campo.
   O delta deve ser **localizado** no choque. Se vazar pro escoamento livre,
   o filtro está pesado demais.
2. `stability_compare.png` — espectros sobrepostos + modo líder de cada ramo.
   Aqui você decide: o cleanup moveu **só modos espúrios** (bom) ou mexeu
   nos modos físicos dominantes (retroceda).
3. `resolvent_compare.png` — curva de ganhos + modo de resposta líder.
4. `summary.json`:
   - `cleanup.filter.relative_l2_change[field]` → tipicamente `1e-3` a `1e-2`.
     Acima de `1e-1` = achatando o escoamento.
   - Deslocamento dos autovalores e razão de ganhos entre ramos.

Se o espectro cleaned ainda tem modos numéricos, suba `filter.passes`
(1 → 2 → 3) e/ou aumente `sensor.buffer_cells` (3 → 4 → 5).

## 5. Rodando em paralelo

Por default `run.sh` roda serial (rank 1). Para MPI, use as mesmas env vars
do `run_case_once.sh`:

```bash
# Portável — qualquer máquina, sem rankfile
LINSTAB2D_MPI_PROCS=4 LINSTAB2D_MPI_PRESET=none \
  bash results/case2/ducros/run.sh results/case2/ducros/02_sensor_filter.jsonc

# Nesta máquina — 10 ranks pinados no socket 0
LINSTAB2D_MPI_PROCS=10 LINSTAB2D_MPI_PRESET=socket0 \
  bash results/case2/ducros/run.sh results/case2/ducros/02_sensor_filter.jsonc

# Com heartbeat mais frequente pra runs longos
LINSTAB2D_MPI_PROCS=10 LINSTAB2D_MPI_PRESET=socket0 \
LINSTAB2D_STAGE_HEARTBEAT_SECONDS=10 \
  bash results/case2/ducros/run.sh results/case2/ducros/02_sensor_filter.jsonc
```

Presets aceitos: `none`, `socket0`, `socket1`, `socket01`, `custom` (veja
`tutorial.md` §5 para o detalhe sobre rankfiles e portabilidade).

## 6. Schema do bloco `shock_cleanup`

Estrutura completa com defaults (`_DEFAULT_CFG` em `shock_cleanup.py:18-41`):

```jsonc
"shock_cleanup": {
  // Chave mestra. Se false, nada roda (mesmo que sub-blocos estejam true).
  "enabled": false,

  "sensor": {
    "enabled": false,
    // "compression_ducros" (padrão) ou "compression_only".
    // "modified_ducros" é reservado — não use.
    "kind": "compression_ducros",

    // div <= compression_threshold aciona a máscara. -1.0 = desligado.
    // Valores típicos quando ligado: -3.0 a -5.0.
    "compression_threshold": -1.0,

    // div²/(div²+vort²) >= ducros_threshold aciona. Default 0.95.
    // Menor = mais agressivo. Maior = só choques bem fortes.
    "ducros_threshold": 0.95,

    "eps": 1e-14,

    // Dilata a máscara por N células — margem de segurança.
    "buffer_cells": 3,

    // Gaussiana sobre preserve_weight. 0.0 = transição abrupta.
    "blend_sigma": 0.0
  },

  "filter": {
    "enabled": false,
    // Passadas do filtro binomial 5-pontos.
    "passes": 1,
    // ["x"], ["y"] ou ["x","y"]. Use ["y"] só se x oversuavizar.
    "directions": ["x", "y"],
    // Campos a filtrar. "w" costuma não importar em 2D.
    "fields": ["rho", "u", "v", "T"]
  },

  "sponge": {
    "enabled": false,
    // Amplitude máxima do sponge na região de choque.
    "amplitude": 0.0,
    // "add": soma ao sponge existente. "max": pega o máximo.
    "mode": "add",
    // Gaussiana sobre o sponge antes de aplicar.
    "smooth_sigma": 0.0
  }
}
```

Regra validada em `shock_cleanup.py:355`: se `filter.enabled` ou
`sponge.enabled` for `true`, `sensor.enabled` tem que ser `true` também.

## 7. Solução de problemas

**`shock_cleanup.sensor.enabled must be true when filter or sponge cleanup is enabled`.**
Ative o sensor — é ele que gera a máscara.

**Runner reclama que `base_config` não encontrado.** Os configs aqui usam
`base_config: "../parity_config.jsonc"` (sobe um nível). Se você copiou o
JSONC pra outro lugar, ajuste o caminho.

**Máscara pega cisalhamento sem choque.** Suba `ducros_threshold` (0.97-0.99).

**Máscara vazia no `cleanup_mask.png`.** Baixe `ducros_threshold` (0.85-0.90)
e/ou ative `compression_threshold` (`-3.0` a `-5.0`).

**`baseflow_compare.png` mostra delta fora do choque.** Filtro vazou: reduza
`passes`, aperte a máscara (thresholds mais seletivos, `buffer_cells` menor).

**Espectro cleaned muito diferente do raw em modos dominantes.** Cleanup
está mexendo na física. Retroceda: reduza `passes`, desligue sponge, aperte
a máscara.

**Script trava em "importing mpi4py".** Clássico singleton hang do MPICH.
Neste wrapper, `LINSTAB2D_MPI_PROCS=1` roda sem `mpiexec`. Se quiser forçar
MPI de verdade, use pelo menos
`LINSTAB2D_MPI_PROCS=2 LINSTAB2D_MPI_PRESET=none`.

## 8. Resumo rápido

```bash
# Etapa 1 — calibrar sensor visualmente (sem filtro)
bash results/case2/ducros/run.sh results/case2/ducros/01_sensor_only.jsonc
# → olhe results/case2/ducros/output/01_sensor_only/cleanup_mask.png

# Etapa 2 — ligar filtro e comparar espectros
bash results/case2/ducros/run.sh results/case2/ducros/02_sensor_filter.jsonc
# → olhe results/case2/ducros/output/02_sensor_filter/stability_compare.png

# Com 4 ranks MPI portáveis
LINSTAB2D_MPI_PROCS=4 LINSTAB2D_MPI_PRESET=none \
  bash results/case2/ducros/run.sh results/case2/ducros/02_sensor_filter.jsonc

# Forçar recompute — apague o output_dir inteiro (cada etapa usa o seu próprio)
rm -rf results/case2/ducros/output/02_sensor_filter
```
