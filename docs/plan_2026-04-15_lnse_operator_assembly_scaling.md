# Plano: LNSE Operator Assembly Scaling

Date: 2026-04-15

## Objetivo

Acelerar a montagem do operador LNSE para casos grandes, com prioridade em:

- reduzir tempo de assembly;
- reduzir picos de memoria durante a montagem;
- preparar o caminho para casos muito grandes, onde a matriz e os temporarios
  passam de dezenas de GB e podem chegar a centenas de GB ou escala de TB;
- manter a camada de solver PETSc/SLEPc como esta, concentrando o trabalho no
  miolo de montagem do operador.

## Estado Atual

Hoje o caminho principal de montagem passa por:

- `src/linstab2d/operators/assembly.py`
- `src/linstab2d/operators/coefficients.py`
- `src/linstab2d/mesh/mesh.py`
- `src/linstab2d/utils/petsc_core.py`

Pontos principais observados:

1. O default dos casos grandes ainda cai em montagem `scipy` quando
   `assembly_backend` nao e definido.
2. Mesmo no backend `petsc`, a montagem atual ainda depende bastante de
   estruturas intermediarias em SciPy.
3. O mesh constroi operadores diferenciais globais em SciPy antes da assembly.
4. A etapa de coefficients materializa muitos tensores `(5, 5, ngp)` complexos.
5. A montagem PETSc atual ainda nao e um caminho distribuido "nativo" de ponta
   a ponta; ela ainda monta blocos CSR em Python/SciPy antes de inserir no
   `PETSc.Mat`.

## Conclusao Tecnica

Para este codigo, "mais rapido possivel" nao significa automaticamente
"montagem direta em PETSc" no estado atual.

O que realmente importa:

- para casos medios/grandes que ainda cabem com folga em memoria, o caminho
  SciPy pode continuar competitivo se os temporarios e reconversoes forem
  reduzidos;
- para casos realmente grandes, a direcao correta e montagem distribuida por
  linhas locais possuidas, sem matriz global SciPy no meio.

Em outras palavras:

- "PETSc direto" so vale a pena como objetivo final se for PETSc distribuido de
  verdade;
- o backend `petsc` atual ainda nao entrega essa arquitetura completa;
- ha ganhos simples e de baixo risco que devem ser feitos antes da mudanca
  estrutural maior.

## Escopo

Em escopo:

- montagem do operador temporal LNSE;
- preparacao de coefficients;
- montagem dos blocos esparsos;
- aplicacao de `inv(B)` e sponge dentro da assembly;
- preallocation e insercao no backend PETSc.

Fora de escopo nesta fase:

- mudancas no solver PETSc/SLEPc;
- tuning de eigensolve ou resolvent;
- alteracao do modelo fisico;
- mudancas de interface publica para usuarios finais, exceto se necessario e
  explicitamente aprovado.

## Impacto Esperado

Mudancas internas na assembly vao afetar quase todos os casos que usam o miolo
LNSE comum:

- `results/case1`
- `results/case2`
- `validation/song_2023`
- `examples/example_1_blasius.py`
- `examples/example_2_jet.py`
- `examples/example_5_paraboloid.py`
- partes de `examples/example_3_streaky_bl.py`

Casos e testes mais sensiveis:

- `examples/example_4_cavity.py`
  porque hoje ele permanece preso ao caminho `scipy`;
- `tests/test_operators.py`
- `tests/test_lnse_verification.py`
- `tests/test_symbolic_equation_parity.py`
- `tests/test_solvers.py`

Conclusao sobre impacto:

- sim, a refatoracao afeta internamente quase todos os casos;
- nao, ela nao precisa quebrar exemplos nem mudar a interface publica se for
  feita por fases;
- o maior risco esta em testes que inspecionam coefficients ou comparam parity
  estrutural entre backends.

## Fase 0: Instrumentacao

Objetivo:
medir exatamente onde o tempo e a memoria estao indo, e gerar um baseline
quantitativo reutilizavel pelas fases seguintes.

Adicionar timings separados para:

- `prepare_coefficients`
- `assemble_raw_blocks`
- `apply_invB`
- `apply_sponge`
- `apply_bc`
- `scipy_to_petsc`
- `petsc_compaction` quando aplicavel

Adicionar tambem estatisticas simples:

- `ngp`
- `ndof`
- `nnz`
- pico de RSS por etapa, quando possivel
- tamanho estimado dos tensores de coefficients

Deliverable concreto:

- as metricas sao emitidas como JSON sidecar em
  `results/<case>/assembly_metrics_<tag>.json` por rank, alem do log textual;
- um script `tools/parity/assembly_metrics_diff.py` le dois sidecars (antes /
  depois) e imprime delta relativo por etapa, para uso no gating das fases
  seguintes.

Casos de referencia para o baseline (rodar ao final da Fase 0 e congelar
numeros):

- `examples/example_1_blasius.py` (pequeno);
- `examples/example_4_cavity.py` (medio, ainda no caminho scipy);
- um caso grande representativo de `results/case2` (grande).

Motivo:
hoje o log junta muita coisa em `assemble_operator_bc`, o que dificulta saber
qual mudanca realmente paga, e sem baseline as fases seguintes nao tem criterio
objetivo de promocao.

## Fase 1: Otimizacoes de Baixo Risco

Objetivo:
ganhar tempo e memoria sem mudar arquitetura.

Itens:

1. Cachear os operadores diferenciais em CSR complex uma vez por montagem.
   Hoje ha reconversoes repetidas em `_build_block_csr(...)`.

2. Cachear `row_nnz` dos operadores derivativos para preallocation.
   Hoje `_block_row_nnz(...)` recalcula `getnnz(axis=1)` repetidamente.

3. Evitar somas esparsas incrementais desnecessarias.
   Revisar o padrao atual de:
   `block = block + diag(coeff) @ operator`
   para reduzir alocacoes intermediarias.

4. Reaproveitar estruturas derivativas entre os 25 blocos, em vez de refazer
   coercao para CSR dentro de cada chamada.

Resultado esperado:

- menor overhead de Python;
- menor numero de alocacoes esparsas temporarias;
- melhoria imediata no caminho SciPy e no backend PETSc atual.

Criterio de aceitacao (vs baseline Fase 0):

- reducao >= 20% em `assemble_raw_blocks` em `example_4_cavity`;
- reducao >= 10% em `assemble_raw_blocks` no caso grande de `results/case2`;
- nenhuma regressao em `prepare_coefficients` ou em paridade numerica.

Se os alvos nao forem atingidos, parar e reavaliar antes de ir para a Fase 2.

## Fase 2: Reducao de Temporarios em Coefficients

Objetivo:
atacar o maior desperdicio de memoria da assembly.

Problema atual:

- os coefficients sao empacotados em varios tensores `(5, 5, ngp)` complexos;
- depois sao copiados de novo para a ordem fisica final;
- isso multiplica a memoria temporaria por rank.

Itens:

1. Preencher diretamente a ordem fisica final, evitando o fluxo
   `coeff_blocks -> result`.

2. Avaliar tornar `B` opcional ou "lazy" no caminho de producao.
   O operador final usa `inv(B)` analitico na assembly, entao `B` completo pode
   ser mantido apenas para testes/parity/export quando necessario.

3. Revisar dtypes e garantir que arrays reais nao sejam promovidos para complex
   antes da hora.

Resultado esperado:

- reducao forte do pico de memoria;
- menor pressao de allocator;
- mais espaco para malhas maiores sem tocar no solver.

Pre-requisito:

- a harness de paridade (ver secao "Validacao Necessaria") precisa estar em
  uso antes de tocar em `compute_cartesian_coefficients` /
  `compute_cylindrical_coefficients`, porque `tests/test_symbolic_equation_parity.py`
  inspeciona a forma/ordem dos tensores `(5, 5, ngp)` e provavelmente vai
  precisar ser atualizado junto com a mudanca.

Sobre tornar `B` opcional/lazy:

- auditar antes os consumidores de `B` fora da assembly, em especial
  `build_lnse_system(...)`, caminhos de resolvent/forcing e qualquer export
  usado por testes;
- sem essa auditoria, manter `B` materializado por padrao.

Criterio de aceitacao (vs baseline Fase 0):

- reducao >= 30% no pico de RSS durante `prepare_coefficients` no caso grande;
- paridade numerica preservada em todos os exemplos da secao "Impacto Esperado".

## Fase 3: Backend PETSc Atual Mais Eficiente

Objetivo:
melhorar o caminho PETSc existente sem ainda exigir refatoracao total.

Dependencia explicita: o item 1 depende dos caches CSR introduzidos na Fase 1;
nao iniciar antes.

Itens:

1. No caminho local/sequencial, evitar `setValues` linha a linha quando houver
   alternativa por bloco/CSR.

2. No caminho distribuido, reduzir montagem de blocos globais antes do slice
   local. O ideal e aproximar o fluxo de "constroi apenas o que o rank possui".

3. Revisar preallocation diagonal/offdiag com estimativas mais proximas do
   padrao real por linha.

Resultado esperado:

- backend `petsc` menos penalizado por overhead Python;
- caminho mais coerente para futuras malhas grandes.

Criterio de aceitacao:

- backend `petsc` chega a <= 1.2x do tempo do caminho scipy no caso medio
  (hoje a diferenca e maior);
- preallocation final tem <= 5% de waste vs nnz efetivo.

## Fase 4: Montagem Distribuida de Verdade

Objetivo:
habilitar casos muito grandes com melhor escalabilidade de memoria.

Criterio de entrada (gate):

Somente iniciar Fase 4 se, apos Fases 1-3, pelo menos uma das condicoes abaixo
ainda nao for atendida no caso-alvo de `results/case2`:

- pico de RSS de assembly <= 50% da RAM disponivel por no;
- tempo total de assembly <= 10% do tempo total do eigensolve.

Se ambas ja forem atendidas, a Fase 4 fica em standby e so e reativada quando
surgir um caso novo que as viole.

Meta arquitetural:

cada rank monta apenas suas linhas locais do operador final, sem montar antes
uma matriz global SciPy completa e sem depender de um CSR global por rank.

Itens:

1. Reestruturar a assembly para operar por ownership range.

2. Gerar coefficients e blocos apenas para linhas locais, ou pelo menos em
   fatias locais grandes e controladas.

3. Inserir diretamente no `PETSc.Mat` distribuido usando blocos locais.

4. Eliminar o caminho onde uma matriz SciPy global e depois convertida para
   PETSc distribuido.

Resultado esperado:

- caminho realmente compativel com casos gigantes;
- menor replicacao de memoria entre ranks;
- base correta para escalar para centenas de GB.

## Ordem Recomendada de Implementacao

1. Fase 0: instrumentacao.
2. Fase 1: caches e remocao de reconversoes.
3. Fase 2: reduzir temporarios em coefficients.
4. Reavaliar ganho real com malhas medias e grandes.
5. Fase 3: melhorar backend PETSc atual.
6. Fase 4: montagem distribuida nativa por linhas locais.

Motivo:

- Fases 1 e 2 tem melhor relacao risco/retorno.
- Fase 4 e a mais importante para o futuro, mas tambem a mais invasiva.

## Compatibilidade e Risco

Risco baixo:

- instrumentacao;
- cache de CSR;
- cache de `row_nnz`;
- remocao de reconversoes repetidas;
- separar timings por etapa.

Risco medio:

- mudar internals de `compute_cartesian_coefficients(...)`;
- mudar internals de `compute_cylindrical_coefficients(...)`;
- tornar `B` opcional no caminho de producao.

Risco alto:

- mudar o modelo mental da assembly PETSc para ownership local;
- trocar default global de backend cedo demais;
- alterar retorno publico de `LNSECoefficients` sem camada de compatibilidade.

Nota especifica sobre testes:
`tests/test_symbolic_equation_parity.py` quase certamente depende da forma e
ordem atuais dos tensores `(5, 5, ngp)` retornados por
`compute_cartesian_coefficients` / `compute_cylindrical_coefficients`. A Fase 2
vai forcar sua atualizacao. Tratar a atualizacao como parte da entrega da fase,
nao como "conserto posterior".

## Politica de Compatibilidade

Durante as fases 1 a 3:

- preservar assinaturas publicas;
- preservar `build_lnse_operator(...)`;
- preservar `build_lnse_operator_petsc(...)`;
- preservar `build_lnse_system(...)`;
- nao mudar defaults em exemplos/configs sem paridade demonstrada.

Durante a fase 4:

- manter o novo caminho atras de `backend='petsc'`;
- limitar inicialmente ao `model='2D'`;
- manter `axy` no caminho antigo ate validacao especifica.

## Validacao Necessaria

Mesmo sem redefinir o solver, a assembly precisa continuar equivalente.

Verificacoes necessarias:

1. paridade numerica entre backend SciPy e PETSc;
2. paridade de `nnz` e estrutura esparsa;
3. paridade em `2D` e `axy`;
4. estabilidade dos exemplos principais;
5. preservacao dos casos de parity ja existentes.

Nao e necessario mudar todo o suite de uma vez, mas qualquer fase que altere
coefficients ou montagem precisa ser validada nos testes de operator parity.

Harness de paridade (entregavel da Fase 0, usado pelas Fases 1-4):

- estender `tools/parity/compare_martini_examples.py` ou criar
  `tools/parity/assembly_parity.py` comparando, entre "antes" e "depois":
  - `nnz` total e por bloco 5x5;
  - norma Frobenius de `L0_depois - L0_antes`;
  - estrutura esparsa (mesmo padrao de indices, tolerancia zero);
  - autovalores dominantes do eigensolve de referencia, para sanity check.
- o script deve aceitar dois artefatos `.npz` ou um par (scipy, petsc) e
  retornar exit code nao-zero em caso de divergencia acima das tolerancias.
- executar o harness em cada fase que toque `coefficients` ou `assembly`.

## Recomendacao Final

Recomendacao objetiva:

- comecar agora por Fase 0, 1 e 2;
- nao trocar o backend default ainda;
- tratar a montagem PETSc distribuida nativa como a meta estrutural para casos
  realmente grandes;
- so promover esse caminho depois de demonstrar paridade e vantagem real.

## Proxima Acao Sugerida

Implementar primeiro:

1. instrumentacao detalhada da assembly;
2. cache de derivativos CSR complex;
3. cache de `row_nnz`;
4. remocao da duplicacao `coeff_blocks -> result` nos coefficients.

Esses passos ja devem produzir uma leitura mais clara do custo real e ganhos
concretos sem alterar o comportamento externo dos casos.
