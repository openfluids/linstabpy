# Tutorial de Execução do Case 2

Este é o guia prático para quem quer rodar `results/case2/` sem ter que
adivinhar o fluxo. O objetivo aqui é explicar:

- como rodar **um caso só**
- como rodar **uma campanha com vários casos**
- quais arquivos editar
- como usar MPI sem lançar vários casos independentes por engano
- como manter o caso portátil para outra máquina, inclusive uma máquina mais
  simples com 1 socket

Todos os comandos abaixo partem da raiz do repositório (`linstabpy/`).

## 1. Regra de bolso

Se você estiver em dúvida, use estas regras:

- quer rodar **um caso só**: use `run_case_once.sh`
- quer rodar **vários casos com CSV/JSON final**: use `perf_memory.py`
- quer portabilidade para outra máquina: use `LINSTAB2D_MPI_PRESET=none`
- quer um teste rápido seguro: use `LINSTAB2D_MPI_PROCS=1`
- só use `socket0`, `socket1` ou `socket01` se souber que os `rankfile`s desta
  pasta batem com a máquina atual

## 2. Dois modos de uso

### A. Rodar um caso só

Arquivos principais:

- [case2.py](case2.py)
- [run_case_once.sh](run_case_once.sh)
- [parity_config.jsonc](parity_config.jsonc)

Esse é o caminho certo quando você quer:

- testar um JSONC novo
- mudar um parâmetro e rodar de novo
- depurar `omega_target`, `stride`, `ncv`, `nev`, MPI ou filtros
- gerar um único `.npz`

### B. Rodar uma campanha com vários casos

Arquivos principais:

- [perf_memory.py](perf_memory.py)
- [run_perf_queue_once.sh](run_perf_queue_once.sh)
- um JSONC de campanha, por exemplo [evp_stride2_spectra.jsonc](evp_stride2_spectra.jsonc)

Esse é o caminho certo quando você quer:

- comparar vários `omega_target`
- fazer sweep de `stride`, `ncv`, `eps_tol` ou `nev`
- salvar logs e NPZs organizados por caso
- gerar tabela CSV e resumo JSON

## 3. Arquivos que realmente importam

No caminho ativo do `case2`, pense assim:

- [parity_config.jsonc](parity_config.jsonc)
  Configuração-base completa do caso.
- [case2.py](case2.py)
  Ponto de entrada do solver para um caso.
- [run_case_once.sh](run_case_once.sh)
  Script recomendado para um caso.
- [perf_memory.py](perf_memory.py)
  Orquestrador de campanhas.
- [run_perf_queue_once.sh](run_perf_queue_once.sh)
  Script de fila para campanhas.
- [README.md](README.md)
  Visão geral científica da campanha.
- `output/<campanha>/`
  Logs, NPZs, PNGs e tabelas.
- [socket0.rankfile](socket0.rankfile), [socket1.rankfile](socket1.rankfile), [socket01.rankfile](socket01.rankfile)
  Arquivos de pinagem MPI desta máquina específica.

## 4. O erro de MPI mais comum

Se você quer **um caso com 10 ranks MPI**, o `mpiexec` deve envolver o
**solver** (`case2.py`), não o harness de campanha.

### Certo

Caso portátil, sem depender de `rankfile`:

```bash
LINSTAB2D_MPI_PROCS=4 \
LINSTAB2D_MPI_PRESET=none \
results/case2/run_case_once.sh \
  results/case2/parity_config.jsonc \
  results/case2/output/runs/case2_manual.npz
```

Caso com pinagem local desta máquina:

```bash
LINSTAB2D_MPI_PROCS=10 \
LINSTAB2D_MPI_PRESET=socket0 \
results/case2/run_case_once.sh \
  results/case2/parity_config.jsonc \
  results/case2/output/runs/case2_manual.npz
```

### Errado

Não faça isso se a intenção é rodar **um caso só**:

```bash
mpiexec -n 10 ./.venv/bin/python results/case2/perf_memory.py
```

Isso lança **10 harnesses**, não 1 caso com 10 ranks.

Também é errado fazer loop manual:

```bash
for i in $(seq 1 10); do
  ./.venv/bin/python results/case2/case2.py
done
```

Isso lança **10 execuções seriais independentes**.

## 5. Como rodar um caso só

### Passo 1. Copie a configuração-base

```bash
cp results/case2/parity_config.jsonc results/case2/meu_caso_stride2.jsonc
```

### Passo 2. Edite o novo JSONC

Os grupos de parâmetros mais importantes são:

- `paths.case_dir`
  Diretório do baseflow. No caso ativo, normalmente é `baseflow`.
- `numerics.stride_ksi`, `numerics.stride_eta`
  Coarsening da malha.
  `1` = resolução cheia, `2` = campanha ativa, `4` = legado exploratório.
- `numerics.fd_order`
  Ordem das diferenças finitas.
- `numerics.beta`
  Número de onda spanwise. `0.0` = modo uniforme em `z`.
- `numerics.alpha_filter`
  Filtro de malha. Para EVP, hoje o padrão recomendado é `"none"`.
- `analysis.run_stability`
  Liga ou desliga EVP.
- `analysis.run_resolvent`
  Liga ou desliga resolvent.
- `analysis.n_modes_stability`
  Quantos autovalores/autovetores pedir ao SLEPc.
- `analysis.omega_target`
  Alvo do shift-invert no plano complexo.
  Convenção do Martini: `real = omega_r`, `imag = omega_i`.
- `solver.eigen.st_solver`
  Backend do shift-invert. Hoje usamos `"mumps"`.
- `solver.eigen.eps_tol`
  Tolerância do EPS.
- `solver.eigen.ncv`
  Tamanho do subespaço de Krylov/Arnoldi.
- `solver.eigen.conv_test`
  Teste de convergência do EPS.
- `solver.eigen.true_residual`
  Pede resíduos “verdadeiros”.
- `solver.eigen.eps_max_it`
  Máximo de iterações do EPS.

Se `analysis.omega_target` não for definido, o runtime usa:

```text
omega_target = 2*pi*St*ratio_uinf/ratio_l
```

Para estudos de EVP, vale a pena explicitar `omega_target` no JSONC.

### Passo 3. Rode usando o wrapper

Teste rápido serial:

```bash
LINSTAB2D_MPI_PROCS=1 \
results/case2/run_case_once.sh \
  results/case2/meu_caso_stride2.jsonc \
  results/case2/output/runs/meu_caso_stride2.npz
```

Execução paralela portátil numa máquina com 1 socket:

```bash
LINSTAB2D_MPI_PROCS=4 \
LINSTAB2D_MPI_PRESET=none \
results/case2/run_case_once.sh \
  results/case2/meu_caso_stride2.jsonc \
  results/case2/output/runs/meu_caso_stride2.npz
```

Com heartbeat:

```bash
LINSTAB2D_MPI_PROCS=4 \
LINSTAB2D_MPI_PRESET=none \
LINSTAB2D_STAGE_HEARTBEAT_SECONDS=30 \
results/case2/run_case_once.sh \
  results/case2/meu_caso_stride2.jsonc \
  results/case2/output/runs/meu_caso_stride2.npz
```

### Passo 4. Veja onde saiu

No modo “um caso só”, o principal artefato é o `.npz` que você passou:

```text
results/case2/output/runs/meu_caso_stride2.npz
```

## 6. Como escolher o MPI no wrapper

O wrapper [run_case_once.sh](run_case_once.sh) usa estas variáveis:

- `LINSTAB2D_MPI_PROCS`
  Número de ranks MPI.
- `LINSTAB2D_MPI_PRESET`
  Preset do comando MPI.

Valores aceitos por `LINSTAB2D_MPI_PRESET`:

- `none`
  Usa `mpiexec` genérico, sem `rankfile`.
  Este é o caminho recomendado para máquinas mais simples e portáveis.
- `socket0`
  Usa [socket0.rankfile](socket0.rankfile).
- `socket1`
  Usa [socket1.rankfile](socket1.rankfile).
- `socket01`
  Usa [socket01.rankfile](socket01.rankfile).
- `custom`
  Usa a string em `LINSTAB2D_MPI_LAUNCHER`.

Exemplos:

Teste rápido serial:

```bash
LINSTAB2D_MPI_PROCS=1 \
results/case2/run_case_once.sh \
  results/case2/meu_caso_stride2.jsonc \
  results/case2/output/runs/meu_caso_stride2.npz
```

4 ranks numa máquina simples com 1 socket:

```bash
LINSTAB2D_MPI_PROCS=4 \
LINSTAB2D_MPI_PRESET=none \
results/case2/run_case_once.sh \
  results/case2/meu_caso_stride2.jsonc \
  results/case2/output/runs/meu_caso_stride2.npz
```

10 ranks no `socket0` desta máquina:

```bash
LINSTAB2D_MPI_PROCS=10 \
LINSTAB2D_MPI_PRESET=socket0 \
results/case2/run_case_once.sh \
  results/case2/meu_caso_stride2.jsonc \
  results/case2/output/runs/meu_caso_stride2.npz
```

Comando MPI customizado:

```bash
LINSTAB2D_MPI_PROCS=4 \
LINSTAB2D_MPI_PRESET=custom \
LINSTAB2D_MPI_LAUNCHER="mpiexec --bind-to core" \
results/case2/run_case_once.sh \
  results/case2/meu_caso_stride2.jsonc \
  results/case2/output/runs/meu_caso_stride2.npz
```

Observação importante:

- os `rankfile`s desta pasta são específicos da máquina atual
- em outra máquina, comece com `LINSTAB2D_MPI_PRESET=none`
- só mude para `custom` se realmente precisar de um launcher específico

## 7. Máquina simples com 1 socket

Se outra pessoa for rodar isso em uma máquina simples, a recomendação é:

1. não usar `rankfile`
2. começar com poucos ranks
3. validar primeiro um caso só

Exemplo:

```bash
LINSTAB2D_MPI_PROCS=4 \
LINSTAB2D_MPI_PRESET=none \
LINSTAB2D_STAGE_HEARTBEAT_SECONDS=30 \
results/case2/run_case_once.sh \
  results/case2/parity_config.jsonc \
  results/case2/output/runs/case2_socket_unico.npz
```

Depois disso, subir gradualmente:

- `MPI_PROCS=1` para teste rápido
- `MPI_PROCS=2` para validar MPI
- `MPI_PROCS=4` ou `8` se a máquina aguentar

Não assuma que `10` é sempre o valor certo. Em outra máquina, o valor certo é
o número de cores que você quer dedicar ao caso, não o número usado neste host.

## 8. Como saber se o caso está vivo

Para runs longos, use:

```bash
LINSTAB2D_STAGE_HEARTBEAT_SECONDS=30
```

Isso faz o solver imprimir linhas do tipo:

- `stage heartbeat: stage=stability elapsed=...`
- `EPS iter=... nconv=...`

Se quiser monitor mais verboso do PETSc/SLEPc:

```bash
PETSC_OPTIONS="-eps_monitor -log_view"
```

Exemplo:

```bash
LINSTAB2D_MPI_PROCS=4 \
LINSTAB2D_MPI_PRESET=none \
LINSTAB2D_STAGE_HEARTBEAT_SECONDS=30 \
PETSC_OPTIONS="-eps_monitor -log_view" \
results/case2/run_case_once.sh \
  results/case2/meu_caso_stride2.jsonc \
  results/case2/output/runs/meu_caso_stride2.npz \
  2>&1 | tee meu_caso_stride2.log
```

## 9. Como rodar uma campanha com vários casos

Uma campanha é um JSONC com três blocos:

- `outputs`
  Onde salvar resumo, logs e NPZs.
- `defaults`
  O que vale para todos os casos.
- `cases`
  A lista de variações.

Exemplos ativos:

- [evp_stride2_spectra.jsonc](evp_stride2_spectra.jsonc)

Exemplos históricos:

- [evp_stride2_stationary_probe.jsonc](legacy/evp_stride2_stationary_probe.jsonc)
- [evp_stride2_stationary_probe_nev6.jsonc](legacy/evp_stride2_stationary_probe_nev6.jsonc)
- [evp_stride4_localize.jsonc](legacy/evp_stride4_localize.jsonc)

Estrutura mínima de um JSONC de campanha:

```jsonc
{
  "outputs": {
    "report_json": "output/minha_campanha/latest.json",
    "report_csv": "output/minha_campanha/table.csv",
    "log_dir": "output/minha_campanha/logs",
    "run_dir": "output/minha_campanha/runs"
  },
  "defaults": {
    "base_config": "parity_config.jsonc",
    "python_exe": "../../.venv/bin/python",
    "launcher": {
      "mpi_procs": 4,
      "launcher": "mpiexec --bind-to core"
    },
    "case_defaults": {
      "config_overrides": {
        "analysis": {
          "run_stability": true,
          "run_resolvent": false
        }
      }
    }
  },
  "cases": [
    {
      "case_id": "caso_a",
      "enabled": true,
      "config_overrides": {
        "analysis": {
          "omega_target": { "real": 0.0, "imag": -0.85 }
        }
      }
    }
  ]
}
```

Para rodar a campanha:

```bash
LINSTAB2D_CASE2_PERF_CONFIG=results/case2/evp_stride2_spectra.jsonc \
./.venv/bin/python results/case2/perf_memory.py
```

Importante:

- **não** rode `perf_memory.py` dentro de `mpiexec`
- o próprio harness já monta o comando MPI de cada caso
- cada item de `cases` vira um caso separado com log, config snapshot e NPZ
- em outra máquina, a primeira coisa a revisar é `defaults.launcher`

## 10. Quais arquivos eu preciso criar?

Se quer rodar um caso só:

- crie um JSONC do caso
- defina um caminho de saída para o `.npz`

Na prática:

- copie [parity_config.jsonc](parity_config.jsonc)
- edite
- rode com [run_case_once.sh](run_case_once.sh)

Se quer rodar uma campanha:

- crie um JSONC de campanha
- opcionalmente use outro JSONC-base além de `parity_config.jsonc`

Na prática:

- copie um dos `evp_*.jsonc`
- ajuste `outputs`, `defaults`, `cases`
- rode com [perf_memory.py](perf_memory.py)

## 11. Como plotar depois

Espectro salvo de uma campanha:

```bash
./.venv/bin/python results/case2/plot_saved_spectra.py \
  --run-dir results/case2/output/evp_stride2_spectra/runs
```

Modos salvos:

```bash
./.venv/bin/python results/case2/plot_saved_modes.py \
  --run-dir results/case2/output/evp_stride2_spectra/runs
```

## 12. Solução de problemas

**O run termina em 2 s e escreve um NPZ minúsculo.**
Provavelmente a saída já existia e foi reaproveitada. Apague o `.npz` de saída
e rode de novo.

**Apareceram 10 logs iguais ou `COMM_WORLD size=1`.**
Comando MPI errado. Use `run_case_once.sh` ou confira se o `mpiexec` usado é
compatível com o `mpi4py`/PETSc do ambiente.

**Roda nesta máquina, mas não roda em outra.**
Provavelmente o problema é `rankfile` absoluto. Na outra máquina, comece com:

```bash
LINSTAB2D_MPI_PRESET=none
```

**Não sei se devo começar por campanha ou caso único.**
Comece sempre por um caso só. Só passe para `perf_memory.py` depois que o caso
único estiver rodando corretamente.
