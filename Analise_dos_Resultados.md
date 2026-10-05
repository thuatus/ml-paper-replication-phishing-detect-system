# Análise dos Resultados — Replicação PHILDER
### Aplicações Distribuídas — PPGEE/UnB | Fase 2

> Este documento registra, em ordem cronológica, cada execução do pipeline de replicação do PHILDER (BISPO, G. D. et al., IEEE Access, 2025) e compara os resultados obtidos com os do artigo original. Para a fundamentação teórica das métricas, a análise crítica da normalização de TES/RTDE na Tabela 4 do artigo e o detalhamento estatístico completo do achado de vazamento de dados (Seções 3.5, 5.5 e 5.6), ver `Embasamento_Teorico_Metricas_PHILDER.md`.

---

## 1. Linha do tempo geral das implementações e medições

| Data | Configuração | O que mudou em relação à execução anterior |
|---|---|---|
| 17/09/2026 | Dataset original — base Nazario com 1.501 amostras de phishing (sem redução à escala do artigo), 3 épocas | Primeira execução completa do pipeline, ponta a ponta |
| 19/09/2026 | Undersampling simulando a escala do artigo original — base Nazario reduzida a 501 amostras, 3 épocas | Redução do phishing a 501 amostras (`amostrar_quantidade_artigo`), para igualar a escala do dataset do artigo (2.501 legítimos + 501 phishing) |
| 21/09/2026 | 501 amostras, **10 épocas** + parâmetros para evitar overfitting (`early stopping`, `load_best_model_at_end`, `weight_decay`, `save_total_limit`) | Aumento do número de épocas (3→10) e adoção de proteções contra overfitting. Precisão de 1,0 chama atenção e leva à investigação de vazamento de dados |
| 22/09/2026 | 501 amostras, 10 épocas + anti-overfitting + **deduplicação (`drop_duplicates`) depois da consolidação do dataset** | Correção do vazamento de dados (diagnóstico completo na Seção 5.5 do Embasamento Teórico) — remove duplicatas de ham e phishing juntas, após consolidar as duas fontes. Efeito colateral: dataset de teste encolhe de 301 para 289 amostras |
| 23/09/2026 | 501 amostras, 10 épocas + anti-overfitting + **deduplicação das amostras de phishing "antes" da consolidação (desenhada)** | Correção desenhada para o efeito colateral da etapa anterior — o pool bruto do Nazario deveria ser deduplicado pelo texto processado *antes* de amostrar os 501 e-mails (`amostrar_quantidade_artigo(..., deduplicar_antes=True)`). **Adotada como referência provisória nesta data — posteriormente descoberto (24/09, ver Seção 1.2) que a célula do notebook nunca chegou a executar essa lógica.** |
| 24/09/2026 | Tentativa de estender a deduplicação prévia também ao lado dos legítimos (SpamAssassin) | A precisão volta a ficar suspeita (~1,0). Investigação revela dois bugs reais: (1) a célula do Módulo A no notebook nunca chamava `amostrar_quantidade_artigo` — usava código residual de 19/09 (`random.sample` sem deduplicação), tornando todas as edições do módulo `.py` entre 21/09–23/09 código morto; (2) ao corrigir isso e estender a deduplicação ao lado dos legítimos, uma troca de variáveis inverteu os rótulos de classe do dataset inteiro. Ambos os bugs corrigidos e validados por log de execução no mesmo dia |
| 25/09/2026 (a) | 501 phishing deduplicado + 2.501 legítimos **sem** deduplicação prévia, 10 épocas + anti-overfitting, pipeline do notebook já corrigido (Bug 1 do Módulo A resolvido) | Primeiro lote realmente executado após a correção do código morto do Módulo A — ainda sem estender a deduplicação ao lado do SpamAssassin. Seis execuções repetidas (arquivos `resultado_final_25092026_estavel[1-6].csv`) |
| 25/09/2026 (b) | 501 phishing + 2.446 legítimos (ambos deduplicados antes da amostragem), 10 épocas + anti-overfitting, pipeline do notebook verificado linha a linha | Extensão da deduplicação prévia também ao lado dos legítimos (após corrigir o Bug 2, de troca de variáveis). Seis execuções repetidas (arquivos `resultado_final_25092026-3_estavel[1-6].csv`) + checagem programática explícita de sobreposição treino/teste (`sobreposição = 0`) confirmam, pela primeira vez de forma direta, a ausência de vazamento. **Resultado de referência definitivo da replicação a partir desta data**, substituindo o de 23/09 |
| 01/10/2026 | Mesmos dados de 25/09(b); **protocolo de treino fiel ao artigo** (`treinar_modelo(..., protocolo="artigo")`): 10 épocas fixas, sem early stopping e sem seleção de checkpoint pelo F1 do teste; avaliação por época mantida só para a curva de perda; sem gravação de checkpoints; tempo de avaliação por época registrado à parte (`tempo_treino_liquido`) | Remove o **viés de seleção de modelo no conjunto de teste** (o lote 25/09(b) usava o teste para decidir a parada e o checkpoint final). Seis execuções repetidas (arquivos `resultado_final-[1-6]_01102026.csv`, curvas `curva_perda-[1-6]_01102026.png`). Ver Seção 3.4 |
| 02/10/2026 | Mesmos dados e protocolo "artigo" de 01/10; **modelo passado como parâmetro** e padrão de saída por modelo (`modelo/<modelo>/`, `resultados/<modelo>/`) | **Expansão para os 5 modelos do artigo** (TinyBERT, MiniLM, DistilBERT, MobileBERT, ALBERT) — rodada-piloto, 1 execução por modelo, para avaliar viabilidade, tempos e resultados. TinyBERT reexecutado com o código novo como validação. Ver Seção 7 |
| 03/10/2026 | Mesmo código, dados e protocolo; execução no **Google Colab** (CPU, 2 núcleos virtuais AMD EPYC 7B12, sem GPU; Linux) | **Reprodutibilidade entre ambientes.** Descoberta e correção de dependência da ordem de leitura dos arquivos do SpamAssassin (`glob` sem ordenação: alfabética no Windows, arbitrária no Linux), que gerava outro dataset com as mesmas sementes. Após a correção (`sorted(glob.glob(...))`, aplicada no Colab e, no mesmo dia, no código do PC local), seis execuções do TinyBERT reproduzem as métricas preditivas do PC local. Ver Seção 8 |
| 03–04/10/2026 | Google Colab (mesmo ambiente da Seção 8), 1 execução de cada um dos 5 modelos | **Expansão no Colab**: 4 dos 5 modelos repetem exatamente a acurácia do piloto local; o ALBERT passa de 9 para 3 erros — evidência de **instabilidade de fine-tuning** (platô inicial em ln 2 no PC, ausente no Colab). Na CPU de 2 núcleos, a vantagem de tempo do TinyBERT cresce (até 18× no treino e 23× na inferência). Ver Seções 8.7–8.9 |

### 1.1 Nota sobre "n (teste)": não é uma contagem só de phishing

Em todas as tabelas deste documento e do Embasamento Teórico, **"n (teste)" é o tamanho total do conjunto de teste — as duas classes somadas, não uma contagem isolada de e-mails de phishing.** O artigo descreve a divisão como *"data were then split into 70% training and 30% testing sets using the datasets library from Hugging Face"*, e essa proporção é aplicada sobre o dataset **já balanceado pelo undersampling** (Módulo C), não sobre o dataset bruto/desbalanceado:

$$501\ \text{phishing} + 501\ \text{legítimos} = 1.002\ \text{registros} \quad\Rightarrow\quad 1.002 \times 0{,}30 = 300{,}6 \approx 301$$

Ou seja, os "301" (ou "289", na execução de 22/09 com deduplicação pós-consolidação) que aparecem nas tabelas abaixo são o **total** do conjunto de teste, não 301 e-mails de phishing. Como o split de treino/teste não é estratificado por classe (é uma amostragem aleatória sobre um dataset que já está balanceado 50/50), o teste sai próximo de metade/metade, mas não exatamente: na execução de 21/09, por exemplo, a matriz de confusão reconstruída mostra 150 phishing + 151 legítimos = 301 (ver Seção 5.5 do Embasamento Teórico) — o que também bate com a checagem direta feita no lote de referência de 25/09(b) (`Teste: {0: 151, 1: 150}`).

### 1.2 Nota sobre a mudança de resultado de referência: de 23/09 para 25/09

As execuções de 21/09 a 23/09 documentam corretamente o **processo de detecção** do vazamento de dados (precisão suspeita → checagem de duplicidade → correção desenhada). Porém, em 24/09 foi descoberto que a célula do notebook responsável pelo Módulo A nunca havia sido atualizada para de fato chamar `amostrar_quantidade_artigo` — ela mantinha um trecho de código residual de 19/09 que amostrava o phishing manualmente, sem qualquer deduplicação, e alimentava a consolidação do dataset diretamente com as listas brutas. Isso significa que a "correção" descrita para 23/09 nunca chegou a ser executada como tal; o pipeline que de fato gerou aquele resultado não pôde ser confirmado com certeza.

Corrigido esse bug, o dia 25/09 passou a rodar, pela primeira vez, o pipeline com a deduplicação de fato em execução — mas em **duas configurações distintas**, testadas em sequência no mesmo dia:

- **25/09(a) — dedup apenas do phishing** (`resultado_final_25092026_estavel[1-6].csv`): o Bug 1 (código morto do Módulo A) já estava corrigido, então `amostrar_quantidade_artigo(..., deduplicar_antes=True)` passou a rodar de verdade — mas, nesse ponto, a deduplicação prévia ainda só cobria o pool de phishing (Nazario); o pool de legítimos (SpamAssassin) continuava sendo amostrado sem deduplicação prévia (2.501 legítimos brutos).
- **25/09(b) — dedup de phishing e ham** (`resultado_final_25092026-3_estavel[1-6].csv`): extensão da mesma lógica também ao lado dos legítimos. Essa edição introduziu o Bug 2 (troca de variáveis, rótulos invertidos) — corrigido no mesmo dia — e o lote final, gerado já com o bug corrigido, é o primeiro cuja cadeia de execução foi verificada linha a código e confirmada por uma checagem programática explícita de sobreposição treino/teste (resultado: zero sobreposição, zero duplicatas internas).

Por isso, a partir deste documento, **25/09(b) substitui 23/09 como resultado de referência oficial da replicação**, e 25/09(a) fica registrado como um resultado intermediário válido (também sem vazamento comprovado no phishing, mas com o pool de legítimos ainda não deduplicado) — útil para medir o efeito isolado de estender a deduplicação ao SpamAssassin (Seção 4). Os detalhes completos dos dois bugs, do processo de correção e da validação em três camadas estão na Seção 5.6 do Embasamento Teórico.

---

## 2. Resultados do artigo original (referência)

### Tabela 1 — Métricas de Desempenho Preditivo (PHILDER = TinyBERT (2), execução com undersampling)

| Modelo | Acc. | Prec. | Rec. | F1 | T. Treino (s) | T. Teste (s) | Tempo de Detecção (ms) |
|---|---|---|---|---|---|---|---|
| TinyBERT (2) | 0,9927 | 0,9896 | 0,9960 | 0,9927 | 180,52 | 6,22 | 12,9811 |

> **Correção 01/10/2026:** até esta data, os tempos desta tabela traziam os valores da linha **DistilBERT (2)** da Tabela 3 do artigo (111,83 s / 4,48 s / 14,8841 ms). Substituídos pelos da linha **TinyBERT (2)** (180,52 s / 6,22 s / 12,9811 ms). As métricas preditivas já estavam corretas. A Tabela 4 do artigo reporta, para TinyBERT (undersampling), Acurácia 0,9927 ± 0,0006 e F1 0,9927 ± 0,0006 (3 execuções).

### Tabela 2 — Métricas de Eficiência Computacional (valores brutos, não normalizados)

| Métrica | Fórmula bruta sobre a Tabela 3 do artigo | Valor exibido nas Figuras 3–5 do artigo |
|---|---|---|
| TES | 0,9927 / 180,52 = **0,0055** | 0,0389 |
| IES | 0,9927 / 6,22 = **0,1596** | 1,0788 |
| RTDE | 0,9927 / 0,0129811 = **76,47** | 324,7750 |

> As Figuras 3–5 do artigo não são consistentes com os tempos da sua própria Tabela 3 (ex.: TES = 0,0389 implicaria ≈ 25,5 s de treino, valor ausente da Tabela 3). A partir de 01/10/2026, a comparação desta replicação usa a coluna da Tabela 3 (dados medidos). Outras inconsistências internas do artigo: o texto cita tempo de teste do TinyBERT com undersampling = 2,57 s (Tabela 3: 6,22 s) e "12.9811 s" por amostra (Tabela 3: ms).

> Ver Seção 3.5 do Embasamento Teórico para a discrepância entre esses valores brutos e os valores normalizados/consolidados que o artigo reporta na sua própria Tabela 4 (0,80–0,93) — uma lacuna de reprodutibilidade do artigo original, não desta replicação.

---

## 3. Resultados desta replicação (Fase 2), por execução

### 3.1 Tabela 1 — Métricas de Desempenho Preditivo

**17/09/2026 — Base Nazario, 1.501 amostras, 3 épocas**

| Modelo | Acc. | Prec. | Rec. | F1 |
|---|---|---|---|---|
| philder | 0,9936 | 0,9957 | 0,9915 | 0,9936 |

**19/09/2026 — Base Nazario, 501 amostras, 3 épocas**

| Modelo | Acc. | Prec. | Rec. | F1 | T. Treino (s) | T. Teste (s) | Tempo de Detecção (ms) |
|---|---|---|---|---|---|---|---|
| philder | 0,9867 | 1,0000 | 0,9733 | 0,9865 | 255,8025 | 4,5329 | 15,0595 |

**21/09/2026 — Base Nazario, 501 amostras, 10 épocas + anti-overfitting**

| Modelo | Acc. | Prec. | Rec. | F1 | T. Treino (s) | T. Teste (s) | Tempo de Detecção (ms) |
|---|---|---|---|---|---|---|---|
| philder | 0,9933 | 1,0000 | 0,9867 | 0,9933 | 205,5063 | 2,1591 | 7,1734 |

**22/09/2026 — Base Nazario, 501 amostras, 10 épocas + anti-overfitting + deduplicação `df.drop_duplicates` DEPOIS da consolidação**

| Modelo | Acc. | Prec. | Rec. | F1 | T. Treino (s) | T. Teste (s) | Tempo de Detecção (ms) |
|---|---|---|---|---|---|---|---|
| philder | 0,9792 | 0,9799 | 0,9792 | 0,9792 | 298,1438 | 2,0680 | 7,1556 |

**23/09/2026 — deduplicação "desenhada" ANTES da consolidação (pipeline de execução não confirmado — ver Seção 1.2)**

| Modelo | Acc. | Prec. | Rec. | F1 | T. Treino (s) | T. Teste (s) | Tempo de Detecção (ms) |
|---|---|---|---|---|---|---|---|
| philder | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 263,5980 | 2,3067 | 7,6633 |

**25/09/2026 (a) — dedup apenas do phishing (`resultado_final_25092026_estavel[1-6].csv`) — lote de 6 execuções**

| Execução | Acc. | Prec. | Rec. | F1 | T. Treino (s) | T. Teste (s) | Detecção (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 0,9900 | 1,0000 | 0,9800 | 0,9899 | 206,31 | 2,19 | 7,27 |
| 2 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 206,76 | 2,08 | 6,93 |
| 3 | 0,9900 | 0,9933 | 0,9867 | 0,9900 | 171,44 | 2,40 | 7,98 |
| 4 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 208,98 | 2,07 | 6,88 |
| 5 | 0,9934 | 1,0000 | 0,9867 | 0,9933 | 164,59 | 2,16 | 7,19 |
| 6 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 230,27 | 2,46 | 8,18 |
| **Média ± DP** | **0,9939 ± 0,0033** | **0,9989 ± 0,0027** | **0,9889 ± 0,0054** | **0,9939 ± 0,0033** | **198,06 ± 25,02** | **2,23 ± 0,16** | **7,40 ± 0,55** |

Matrizes de confusão reconstruídas (n=301: 150 phishing + 151 legítimos):

| Execução | VP | FN | FP | VN | Erros |
|---|---|---|---|---|---|
| 1 | 147 | 3 | 0 | 151 | 3 |
| 2 | 149 | 1 | 0 | 151 | 1 |
| 3 | 148 | 2 | 1 | 150 | 3 |
| 4 | 149 | 1 | 0 | 151 | 1 |
| 5 | 148 | 2 | 0 | 151 | 2 |
| 6 | 149 | 1 | 0 | 151 | 1 |

**25/09/2026 (b) — dedup de phishing e ham (`resultado_final_25092026-3_estavel[1-6].csv`), pipeline verificado e sobreposição treino/teste = 0 (resultado de referência definitivo) — lote de 6 execuções**

| Execução | Acc. | Prec. | Rec. | F1 | T. Treino (s) | T. Teste (s) | Detecção (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 389,18 | 4,38 | 14,55 |
| 2 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 343,57 | 3,98 | 13,21 |
| 3 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 275,64 | 3,85 | 12,79 |
| 4 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 344,83 | 4,06 | 13,49 |
| 5 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 355,99 | 3,71 | 12,33 |
| 6 | 0,9967 | 0,9934 | 1,0000 | 0,9967 | 404,24 | 3,55 | 11,79 |
| **Média ± DP** | **0,9967 ± 0,0021** | **0,9978 ± 0,0034** | **0,9956 ± 0,0034** | **0,9967 ± 0,0021** | **352,24 ± 44,90** | **3,92 ± 0,29** | **13,03 ± 0,96** |

Matrizes de confusão reconstruídas (n=301: 150 phishing + 151 legítimos, exceto onde indicado):

| Execução | VP | FN | FP | VN | Erros |
|---|---|---|---|---|---|
| 1 | 149 | 1 | 0 | 151 | 1 |
| 2 | 150 | 0 | 0 | 151 | 0 (perfeita) |
| 3 | 149 | 1 | 0 | 151 | 1 |
| 4 | 149 | 1 | 0 | 151 | 1 |
| 5 | 149 | 1 | 1 | 150 | 2 |
| 6 | 150 | 0 | 1 | 150 | 1 |

Checagem de vazamento (mesma configuração de dados usada nas 6 execuções do lote (b), ver Seção 5.6 do Embasamento Teórico para o código completo):

```
Balanceamento aplicado: {1: 501, 0: 501}
Treino: {1: 351, 0: 350}
Teste: {0: 151, 1: 150}
duplicatas no balanceado: 0 de 1002
duplicatas DENTRO de legítimos: 0
duplicatas DENTRO de phishing: 0
E-mails processados idênticos entre treino e teste: 0
```

**01/10/2026 — protocolo fiel ao artigo: 10 épocas fixas, sem early stopping e sem seleção de checkpoint no teste (`resultado_final-[1-6]_01102026.csv`) — lote de 6 execuções**

Mesmos dados de 25/09(b) (501 + 501 balanceados, dedup de phishing e ham, split estratificado 70/30, `random_state=42`). Única mudança: `treinar_modelo(..., protocolo="artigo")`. Todas as 6 execuções rodaram as 10 épocas completas (880 steps).

| Execução | Acc. | Prec. | Rec. | F1 | T. Treino bruto (s) | T. Aval. por época (s) | T. Treino líquido (s) | T. Teste (s) | Detecção (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 401,06 | 22,24 | 378,82 | 2,22 | 7,36 |
| 2 | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 396,51 | 22,09 | 374,41 | 2,24 | 7,44 |
| 3 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 397,82 | 22,58 | 375,24 | 2,21 | 7,35 |
| 4 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 412,32 | 23,94 | 388,38 | 2,26 | 7,49 |
| 5 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 425,64 | 23,60 | 402,04 | 2,72 | 9,03 |
| 6 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 405,07 | 23,00 | 382,07 | 2,43 | 8,07 |
| **Média ± DP** | **0,9928 ± 0,0014** | **0,9922 ± 0,0027** | **0,9933 ± 0,0000** | **0,9928 ± 0,0013** | **406,40 ± 11,02** | **22,91 ± 0,74** | **383,49 ± 10,41** | **2,34 ± 0,20** | **7,79 ± 0,66** |

Matrizes de confusão reconstruídas (n=301: 150 phishing + 151 legítimos):

| Execução | VP | FN | FP | VN | Erros |
|---|---|---|---|---|---|
| 1 | 149 | 1 | 1 | 150 | 2 |
| 2 | 149 | 1 | 2 | 149 | 3 |
| 3 | 149 | 1 | 1 | 150 | 2 |
| 4 | 149 | 1 | 1 | 150 | 2 |
| 5 | 149 | 1 | 1 | 150 | 2 |
| 6 | 149 | 1 | 1 | 150 | 2 |

> Nota de arquivo: a execução 6 foi gravada com o nome padrão (`resultado_final.csv` / `curva_perda.png`) e deve ser renomeada para `resultado_final-6_01102026.csv` / `curva_perda-6_01102026.png`.

### 3.2 Tabela 2 — Métricas de Eficiência Computacional (valores brutos)

**17/09/2026**

| Métrica | Valor bruto |
|---|---|
| TES | 0,00268 |
| IES | 0,1474 |
| RTDE | 138,3831 |

**19/09/2026**

| Métrica | Valor bruto |
|---|---|
| TES | 0,0039 |
| IES | 0,2177 |
| RTDE | 65,505 |

**21/09/2026**

| Métrica | Valor bruto |
|---|---|
| TES | 0,0048 |
| IES | 0,4601 |
| RTDE | 138,4772 |

**22/09/2026**

| Métrica | Valor bruto |
|---|---|
| TES | 0,0033 |
| IES | 0,4735 |
| RTDE | 136,8492 |

**23/09/2026**

| Métrica | Valor bruto |
|---|---|
| TES | 0,00378 |
| IES | 0,4292 |
| RTDE | 129,1910 |

**25/09/2026 (a) — dedup apenas do phishing — média ± DP das 6 execuções**

| Métrica | Média ± DP |
|---|---|
| TES | 0,005088 ± 0,000664 |
| IES | 0,447961 ± 0,032449 |
| RTDE | 134,8363 ± 9,7670 |

**25/09/2026 (b) — dedup de phishing e ham — média ± DP das 6 execuções (resultado de referência definitivo)**

| Métrica | Média ± DP |
|---|---|
| TES | 0,002872 ± 0,000406 |
| IES | 0,255279 ± 0,018430 |
| RTDE | 76,8516 ± 5,5475 |

**01/10/2026 — protocolo fiel ao artigo — média ± DP das 6 execuções**

| Execução | TES (bruto) | TES (líquido) | IES | RTDE |
|---|---|---|---|---|
| 1 | 0,002477 | 0,002622 | 0,4482 | 134,92 |
| 2 | 0,002497 | 0,002644 | 0,4421 | 133,08 |
| 3 | 0,002497 | 0,002647 | 0,4492 | 135,20 |
| 4 | 0,002409 | 0,002558 | 0,4404 | 132,56 |
| 5 | 0,002334 | 0,002471 | 0,3656 | 110,04 |
| 6 | 0,002452 | 0,002600 | 0,4091 | 123,14 |
| **Média ± DP** | **0,002444 ± 0,000063** | **0,002590 ± 0,000067** | **0,4258 ± 0,0330** | **128,16 ± 9,92** |

TES (bruto) = Acurácia / tempo de `trainer.train()` inteiro (inclui a avaliação por época — comparável ao artigo, que também avalia por época, Figura 6). TES (líquido) = Acurácia / (tempo bruto − soma dos `eval_runtime` registrados em `trainer.state.log_history`).

### 3.3 Tabela consolidada — todas as execuções lado a lado

| Data | Configuração | Acc. | Prec. | Rec. | F1 | T. Treino (s) | T. Teste (s) | Detecção (ms) | TES | IES | RTDE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | Artigo (TinyBERT (2), Tabela 3; eficiência pela fórmula bruta — corrigido em 01/10) | 0,9927 | 0,9896 | 0,9960 | 0,9927 | 180,52 | 6,22 | 12,9811 | 0,0055 | 0,1596 | 76,47 |
| 17/09 | 1.501 amostras, 3 ép. | 0,9936 | 0,9957 | 0,9915 | 0,9936 | 370,75* | 6,74* | 7,18* | 0,00268 | 0,1474 | 138,3831 |
| 19/09 | 501 amostras, 3 ép. | 0,9867 | 1,0000 | 0,9733 | 0,9865 | 255,8025 | 4,5329 | 15,0595 | 0,0039 | 0,2177 | 65,505 |
| 21/09 | 501 amostras, 10 ép. + anti-overfitting | 0,9933 | 1,0000 | 0,9867 | 0,9933 | 205,5063 | 2,1591 | 7,1734 | 0,0048 | 0,4601 | 138,4772 |
| 22/09 | + dedup pós-consolidação | 0,9792 | 0,9799 | 0,9792 | 0,9792 | 298,1438 | 2,0680 | 7,1556 | 0,0033 | 0,4735 | 136,8492 |
| 23/09 | + dedup pré-consolidação desenhada (pipeline não confirmado) | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 263,5980 | 2,3067 | 7,6633 | 0,00378 | 0,4292 | 129,1910 |
| 25/09(a) | dedup apenas do phishing, pipeline corrigido (média de 6 exec.) | 0,9939 ± 0,0033 | 0,9989 ± 0,0027 | 0,9889 ± 0,0054 | 0,9939 ± 0,0033 | 198,06 ± 25,02 | 2,23 ± 0,16 | 7,40 ± 0,55 | 0,005088 ± 0,000664 | 0,447961 ± 0,032449 | 134,8363 ± 9,7670 |
| 25/09(b) | + dedup também dos legítimos, pipeline verificado (**referência**, média de 6 exec.) | 0,9967 ± 0,0021 | 0,9978 ± 0,0034 | 0,9956 ± 0,0034 | 0,9967 ± 0,0021 | 352,24 ± 44,90 | 3,92 ± 0,29 | 13,03 ± 0,96 | 0,002872 ± 0,000406 | 0,255279 ± 0,018430 | 76,8516 ± 5,5475 |
| 01/10 | mesmos dados de 25/09(b), **protocolo fiel ao artigo** (10 ép. fixas, sem seleção no teste) (média de 6 exec.) | 0,9928 ± 0,0014 | 0,9922 ± 0,0027 | 0,9933 ± 0,0000 | 0,9928 ± 0,0013 | 406,40 ± 11,02 | 2,34 ± 0,20 | 7,79 ± 0,66 | 0,002444 ± 0,000063 | 0,4258 ± 0,0330 | 128,16 ± 9,92 |

\* T. Treino, T. Teste e Detecção de 17/09 não constavam na Tabela 1 original dessa data — foram derivados algebricamente a partir da Tabela 2 (TES, IES, RTDE) da mesma data, ex.: T. Treino = Acurácia ÷ TES = 0,9936 ÷ 0,00268 ≈ 370,75 s.

### 3.4 Lote 01/10 — protocolo fiel ao artigo: comparação com 25/09(b) e com o artigo

**Motivação.** No lote 25/09(b), o conjunto de teste era usado durante o treino para três decisões — avaliação por época, parada antecipada (`EarlyStoppingCallback`, paciência 2) e recarga do checkpoint de melhor F1 (`load_best_model_at_end`) — e depois para medir o resultado final. Isso é **vazamento por seleção de modelo**: nenhum e-mail do teste entra nos pesos, mas o teste influencia quais pesos são mantidos, inflando a métrica final (o máximo de várias medidas ruidosas é, em média, maior que o valor verdadeiro). É distinto do vazamento de dados de 21–25/09 e não é detectado pela checagem de sobreposição treino/teste. O artigo não descreve conjunto de validação separado nem seleção de checkpoint; sua Figura 6 mostra o treino completo de 10 épocas com acompanhamento da loss de avaliação — protocolo replicado em 01/10.

**Comparação das métricas preditivas**

| Métrica | Artigo (Tab. 3 / Tab. 4) | 01/10 — protocolo "artigo" (n=6) | 25/09(b) — seleção no teste (n=6) | Diferença 25/09(b) − 01/10 | Welch p | Mann-Whitney p |
|---|---|---|---|---|---|---|
| Acurácia | 0,9927 ± 0,0006 | **0,9928 ± 0,0014** | 0,9967 ± 0,0021 | +0,0039 | 0,005 | 0,009 |
| Precisão | 0,9896 | **0,9922 ± 0,0027** | 0,9978 ± 0,0034 | +0,0055 | 0,012 | 0,009 |
| Recall | 0,9960 | **0,9933 ± 0,0000** | 0,9956 ± 0,0034 | +0,0022 | 0,175 | 0,174 |
| F1 | 0,9927 ± 0,0006 | **0,9928 ± 0,0013** | 0,9967 ± 0,0021 | +0,0039 | 0,005 | 0,010 |

**Comparação de tempos e eficiência**

| Métrica | Artigo (Tab. 3, fórmula bruta) | 01/10 (n=6) | 25/09(b) (n=6) |
|---|---|---|---|
| T. Treino bruto (s) | 180,52 | 406,40 ± 11,02 | 352,24 ± 44,90 |
| T. Treino líquido (s) | — | 383,49 ± 10,41 | — |
| T. Teste (s) | 6,22 | 2,34 ± 0,20 | 3,92 ± 0,29 |
| Detecção (ms/amostra) | 12,9811 | 7,79 ± 0,66 | 13,03 ± 0,96 |
| TES (bruto) | 0,0055 | 0,002444 ± 0,000063 | 0,002872 ± 0,000406 |
| TES (líquido) | — | 0,002590 ± 0,000067 | — |
| IES | 0,1596 | 0,4258 ± 0,0330 | 0,2553 ± 0,0184 |
| RTDE | 76,47 | 128,16 ± 9,92 | 76,84 ± 5,55 |

**Leitura**

1. **Com o protocolo do artigo, a replicação empata com o artigo nas métricas preditivas** (Acurácia e F1 iguais até a 4ª casa decimal). A "vantagem" de 25/09(b) sobre o artigo (+0,40 p.p.) tem o mesmo tamanho do efeito do viés de seleção medido aqui (≈ 0,0039 ≈ 1,2 e-mail em 301), e desaparece quando o teste deixa de participar das decisões de treino.
2. **As 6 execuções não são amostras independentes.** O recall tem desvio-padrão zero: 5 das 6 execuções erram exatamente os mesmos 2 e-mails (1 FP + 1 FN), e a execução 2 erra 1 FP a mais. As curvas de perda diferem entre execuções (não-determinismo do treino), mas o modelo final converge para as mesmas decisões, porque dados, split e semente do `Trainer` (42) são fixos. O ± mede só o não-determinismo de execução, não a incerteza do modelo — os p-valores acima devem ser lidos com essa ressalva (pseudo-replicação). Reforça a pendência de variar `random_state` (split) e `seed` (treino).
3. **No protocolo antigo, a seleção no teste amplificava o ruído:** pequenas diferenças na trajetória mudavam a época escolhida, gerando variância maior (DP 0,0021 vs. 0,0014 na acurácia; resultado perfeito, 1,0000, na execução 2 de 25/09(b)). As curvas de 01/10 mostram o mecanismo — ex.: execução 4 com loss de avaliação ≈ 0,02 na época 3, ponto que o early stopping teria selecionado.
4. **Os 2 e-mails errados por todas as execuções são casos estruturalmente difíceis** — candidatos a análise de erros (rótulo ambíguo ou phishing com aparência legítima).
5. **Tempos.** Com épocas fixas, o tempo de treino fica estável (CV 2,7% vs. 12,7% em 25/09(b)) e o TES torna-se comparável entre execuções e entre modelos. A avaliação por época custa só ≈ 5,6% do treino, não explicando a diferença para o artigo. A diferença de tempo de **teste** entre os lotes (2,34 s vs. 3,92 s) não decorre do protocolo (a inferência é idêntica) — é variação de ambiente entre dias, já observada antes (25/09(a): 2,23 s); por isso, diferenças de IES/RTDE entre lotes não devem ser atribuídas a causas metodológicas. Dentro do próprio lote, as execuções 5 e 6 tiveram teste mais lento (2,72 s e 2,43 s), indicando máquina mais carregada.
6. **Contra o artigo, o padrão de tempos não é compatível com "hardware mais fraco" apenas:** o treino desta replicação é ≈ 2,2× mais lento, mas a inferência é ≈ 2,7× mais rápida (IES e RTDE maiores que os do artigo). Hipótese a investigar: configuração não documentada no artigo — comprimento máximo da sequência (aqui 128), batch de inferência, ou inclusão da tokenização no tempo de teste.

---

## 4. Análise evolutiva

**17/09 → 19/09 (redução da escala do dataset e primeiro sinal do vazamento).** Reduzir o phishing de 1.501 para 501 amostras (para igualar a escala do artigo) derruba o recall (0,9915 → 0,9733) — esperado, menos dados de treino de phishing tende a piorar a detecção da classe minoritária. Mas a precisão já salta para **1,0000** nessa execução, mesmo sem os parâmetros anti-overfitting e com apenas 3 épocas. Em retrospecto, isso é o primeiro sinal do vazamento de dados que só seria formalmente diagnosticado na execução de 21/09 (Seção 5.5 do Embasamento Teórico): reduzir o pool de phishing a exatamente 501 amostras aumenta a proporção do dataset ocupada por conteúdo duplicado (a base do Nazario tem uma taxa de duplicação de ~4%, ver Seção 5.5), tornando mais provável que uma duplicata acabe ao mesmo tempo no treino e no teste. Com 1.501 amostras (17/09), o mesmo número absoluto de duplicatas fica diluído num pool maior, reduzindo a chance de uma dessas duplicatas cair especificamente nos dois lados do split.

**19/09 → 21/09 (mais épocas e proteção contra overfitting).** A precisão continua em 1,0000 — o vazamento não desaparece, porque os parâmetros de anti-overfitting (`early stopping`, `weight_decay`) atacam um problema diferente (o modelo memorizar padrões do próprio conjunto de treino) e não têm como corrigir duplicatas entre treino e teste. É essa persistência do artefato, já com um regime de treino mais cuidadoso, que motivou a investigação registrada na Seção 5.5.

**21/09 → 22/09 (primeira correção do vazamento, efetivamente executada).** Deduplicar o dataset já consolidado resolve o problema relatado — a precisão cai para um valor realista (0,9799, muito próximo do recall) — mas tem um efeito colateral não previsto: o alvo do balanceamento por undersampling passa a ser menor (481 em vez de 501, porque o Nazario perdeu 20 exemplares únicos na deduplicação), encolhendo o dataset de treino/teste e derrubando as quatro métricas preditivas (Acc./F1 de ~0,993 para ~0,979) e o TES (de 0,0048 para 0,0033).

**22/09 → 23/09 (correção do efeito colateral — desenhada, mas não confirmada em execução).** A ideia era deduplicar o pool bruto do Nazario *antes* de amostrar os 501 e-mails, resolvendo o vazamento sem encolher o dataset. O resultado observado (Acc. 0,9900, F1 0,9900, recall 0,9933) foi consistente com essa hipótese e, por isso, foi adotado como referência provisória — mas a Seção 1.2 registra que, em 24/09, se descobriu que a célula do notebook não chamava a função responsável por essa lógica, então o mecanismo real por trás desses números específicos não pôde ser confirmado.

**23/09 → 24/09 (reabertura da investigação: a causa raiz real).** Ao tentar estender a deduplicação prévia também ao lado dos legítimos, a precisão voltou a ficar suspeita. A investigação revelou dois bugs de implementação, não de lógica estatística: (1) a célula do Módulo A no notebook nunca invocava `amostrar_quantidade_artigo` — usava um trecho residual de 19/09 que amostrava o phishing manualmente sem deduplicação, tornando as edições do módulo `.py` feitas entre 21/09–23/09 código morto, nunca executado; (2) corrigido esse primeiro bug, uma segunda edição — para estender a deduplicação aos legítimos — trocou por engano quais listas alimentavam `legitimos_amostrados` e `phishing_amostrados`, invertendo os rótulos de classe do dataset inteiro. Ambos os bugs foram corrigidos e validados por log de execução no mesmo dia (Seção 5.6 do Embasamento Teórico traz o código antes/depois de cada correção).

**24/09 → 25/09(a) (primeiro pipeline realmente executado, ainda só com dedup do phishing).** Corrigido o Bug 1 (código morto do Módulo A), o lote 25/09(a) já roda a lógica pretendida — mas ainda só para o Nazario, já que a extensão da deduplicação ao SpamAssassin ainda não tinha sido implementada nesse ponto. Seis execuções repetidas mostraram Acc. 0,9939 ± 0,0033, Prec. 0,9989 ± 0,0027, Rec. 0,9889 ± 0,0054, F1 0,9939 ± 0,0033 — variância saudável (1 a 3 erros por execução), com os erros concentrados quase todos em falsos negativos do lado do phishing (só uma execução teve 1 falso positivo), padrão semelhante ao observado em 23/09.

**25/09(a) → 25/09(b) (extensão da deduplicação aos legítimos, novo bug e validação definitiva).** Ao estender a mesma lógica de deduplicação prévia também ao pool de legítimos (SpamAssassin), uma segunda edição introduziu o Bug 2 (troca de variáveis, rótulos de classe invertidos) — corrigido no mesmo dia. O lote final, 25/09(b), mostrou variância ainda mais saudável que o (a): Acc. 0,9967 ± 0,0021, Prec. 0,9978 ± 0,0034, Rec. 0,9956 ± 0,0034, F1 0,9967 ± 0,0021, com falsos positivos aparecendo pela primeira vez do lado dos legítimos (não só falsos negativos do lado do phishing, como em todas as execuções anteriores, incluindo 25/09(a)) — evidência adicional contra vazamento sistemático. Uma checagem programática explícita de sobreposição treino/teste confirmou, de forma direta (não apenas inferida da plausibilidade das métricas), a ausência total de vazamento: zero e-mails processados idênticos entre treino e teste, zero duplicatas internas em cada classe. Esse é o primeiro resultado da replicação cuja cadeia de execução foi verificada linha a código, e passa a ser a referência oficial.

**Comparando 25/09(a) com 25/09(b): o efeito de deduplicar também o lado dos legítimos.** As quatro métricas preditivas melhoram de (a) para (b) — Acc. de 0,9939 para 0,9967, Prec. de 0,9989 para 0,9978 (leve queda, mas dentro do desvio-padrão de ambos os lotes), Rec. de 0,9889 para 0,9956, F1 de 0,9939 para 0,9967. O ganho mais consistente é no recall, que sobe de forma mais nítida entre os dois lotes — plausível, já que remover legítimos duplicados do pool de treino da classe majoritária tende a expor o modelo a mais conteúdo distinto de legítimos únicos, reduzindo a chance de o modelo "decorar" um padrão redundante e melhorando sua capacidade de distinguir os phishing verdadeiros. Já nas métricas de eficiência, o padrão é o oposto: TES, IES e RTDE caem de (a) para (b) (por exemplo, RTDE de 134,84 para 76,85), e o tempo médio de treino quase dobra (198,06 s para 352,24 s). Isso **não** pode ser atribuído ao tamanho do dataset — em ambos os lotes o undersampling (Módulo C) equaliza as duas classes ao tamanho da classe minoritária (501, o phishing já deduplicado em ambos os casos), então o dataset balanceado final tem exatamente 1.002 registros e ~301 no teste nos dois lotes, não importa se o pool bruto de legítimos tinha 2.501 ou 2.446 antes do corte. A diferença de tempo é, portanto, mais bem explicada por ruído de medição/ambiente entre as duas rodadas de execução (mesmo padrão de sensibilidade a variações de sistema já discutido na Seção 3.1 do Embasamento Teórico), não por uma diferença real de custo computacional entre as duas técnicas de deduplicação.

---

## 5. Comparação final: resultado de referência (25/09(b)) vs. artigo original

> **Atualização 01/10/2026.** Esta seção foi escrita antes de duas correções: (1) os tempos e as métricas de eficiência do artigo usados abaixo vinham das Figuras 3–5 e de uma linha errada da Tabela 3 (ver Seção 2 — valores corrigidos: TES 0,0055, IES 0,1596, RTDE 76,47); (2) o lote 25/09(b) usa o conjunto de teste para seleção de modelo. A comparação atualizada, com o protocolo fiel ao artigo, está na Seção 3.4. A redefinição formal do resultado de referência fica pendente de decisão.

| Métrica | Artigo (TinyBERT (2)) | Replicação (25/09(b), média de 6 exec.) | Diferença |
|---|---|---|---|
| Acurácia | 0,9927 | 0,9967 ± 0,0021 | +0,40 p.p. |
| Precisão | 0,9896 | 0,9978 ± 0,0034 | +0,82 p.p. |
| Recall | 0,9960 | 0,9956 ± 0,0034 | −0,04 p.p. |
| F1 | 0,9927 | 0,9967 ± 0,0021 | +0,40 p.p. |
| TES | 0,0389 | 0,002872 ± 0,000406 | ≈13,5× menor |
| IES | 1,0788 | 0,255279 ± 0,018430 | ≈4,2× menor |
| RTDE | 324,7750 | 76,8516 ± 5,5475 | ≈4,2× menor |

Nas métricas preditivas, a replicação agora **supera** o artigo original em três das quatro métricas (Acurácia, Precisão e F1), com o Recall praticamente empatado (diferença de 0,04 p.p., dentro do desvio-padrão observado entre execuções). É um resultado mais forte do que o de 23/09 (que ficava ligeiramente abaixo do artigo nas quatro métricas) e também mais forte do que o de 25/09(a) (Seção 3.1) — plausível, já que o lote 25/09(b) é o primeiro a deduplicar de fato os dois lados do dataset (legítimos e phishing) antes da amostragem, com pipeline de execução verificado e sobreposição treino/teste comprovadamente nula.

Nas métricas de eficiência, o gap em relação ao artigo permanece grande e consistente (na faixa de 4× a 13,5×) — como já discutido na Seção 3.5 do Embasamento Teórico, essa diferença é atribuída a hardware (o artigo provavelmente usa GPU; esta replicação roda em CPU) e não representa uma divergência metodológica da implementação. O próprio artigo, aliás, reporta essas métricas de forma inconsistente entre os gráficos de barras (usados aqui como referência) e sua Tabela 4 (numa escala não documentada) — o que reforça a recomendação de comparar sempre os valores brutos, não os normalizados.

---

## 6. Conclusão e próximos passos

O resultado de 25/09(b) é adotado como referência definitiva da replicação: metodologicamente correto (sem vazamento de dados, confirmado via checagem programática explícita de sobreposição treino/teste, não apenas pela plausibilidade das métricas), com ambos os pools do dataset (legítimos/SpamAssassin e phishing/Nazario) deduplicados antes da amostragem, e com o pipeline de execução do notebook verificado linha a código — o primeiro da replicação a atingir esse nível de certeza. O lote 25/09(a), mantido lado a lado na Seção 3, documenta o passo intermediário (dedup só do phishing) e permite isolar o efeito específico de estender a deduplicação também ao lado dos legítimos (Seção 4). A evolução registrada nesta linha do tempo — da detecção inicial de um resultado "bom demais" (21/09), passando por uma correção desenhada mas nunca de fato executada (23/09), até a descoberta dos dois bugs reais de implementação e a comparação entre as duas técnicas de deduplicação validadas em 25/09(a) e (b) — é, em si, material especialmente relevante para a entrega de "análise crítica dos métodos de aferição de desempenho e eficácia" prevista na Fase 2: a lição central é que plausibilidade de métricas não substitui verificação direta do código executado e dos dados resultantes.

Itens em aberto para próximas atualizações deste documento:
- Repetir a execução de referência (25/09(b)) com sementes diferentes de `random_state` (todas as 6 execuções atuais usam a mesma semente/configuração, variando apenas por não-determinismo de treino), para isolar variância de semente de variância de treino.
- Incorporar novos marcos à linha do tempo conforme forem executados.
- A extensão da deduplicação prévia ao lado do SpamAssassin (item que estava em aberto neste documento até a atualização anterior) já foi implementada e validada como parte da correção de 24–25/09 — item concluído, com os dois lotes (a) e (b) documentando o antes/depois dessa extensão.

---

## 7. Expansão dos modelos avaliados (Fase 3 — piloto de 02/10/2026)

### 7.1 Contexto da inclusão dos modelos

O PHILDER não é apenas "TinyBERT com undersampling": a escolha do TinyBERT é a **conclusão de uma comparação** entre cinco variantes leves do BERT — DistilBERT, ALBERT, MiniLM, MobileBERT e TinyBERT (Tabela 2 do artigo) —, avaliadas em desempenho preditivo (Tabelas 3 e 4) e em eficiência computacional (TES, IES e RTDE, Figuras 3–5). A afirmação central do artigo é que o TinyBERT oferece o melhor equilíbrio entre acurácia e custo computacional. Replicar só o TinyBERT confirma o desempenho do modelo escolhido, mas **não testa essa afirmação**, que depende da comparação com os demais.

Havia ainda um motivo específico: a Tabela 3 do artigo é internamente inconsistente com a sua própria conclusão. Pelos tempos ali registrados, MiniLM (103,52 s) e DistilBERT (111,83 s) treinam **mais rápido** que o TinyBERT (180,52 s) — o que daria a eles TES maior (0,0096 e 0,0089 contra 0,0055, pela fórmula bruta) —, apesar de o texto e a Figura 3 apontarem o TinyBERT como o mais eficiente em treino e de o TinyBERT ser a arquitetura menor. Só medindo os cinco modelos no mesmo ambiente e com o mesmo protocolo é possível verificar qual ordenação é a real.

Objetivo desta etapa (piloto): **uma execução por modelo**, para (i) validar que o pipeline funciona com todas as arquiteturas, (ii) obter a ordem de grandeza dos tempos na máquina local (CPU) e planejar as rodadas completas, e (iii) obter uma primeira comparação com as Tabelas 3 e 4 do artigo.

**Checkpoints adotados.** O artigo não informa quais checkpoints pré-treinados utilizou (a Tabela 2 descreve apenas as arquiteturas). Adotou-se, para cada modelo, o checkpoint de referência publicado com o artigo original da arquitetura (referências [24]–[28] da bibliografia):

| Modelo | Checkpoint (Hugging Face) | Camadas × dimensão | Parâmetros (aprox.) | Tokenizador |
|---|---|---|---|---|
| TinyBERT | `huawei-noah/TinyBERT_General_4L_312D` | 4 × 312 | ~14,5 M | WordPiece (vocabulário BERT uncased) |
| MiniLM | `microsoft/MiniLM-L12-H384-uncased` | 12 × 384 | ~33 M | WordPiece (vocabulário BERT uncased) |
| DistilBERT | `distilbert-base-uncased` | 6 × 768 | ~66 M | WordPiece (vocabulário BERT uncased) |
| MobileBERT | `google/mobilebert-uncased` | 24 camadas estreitas (gargalo 128) | ~25 M | WordPiece (vocabulário BERT uncased) |
| ALBERT | `albert-base-v2` | 12 × 768 (camadas com pesos compartilhados) | ~12 M | **SentencePiece** (próprio) |

> **Desvio documentado — tokenizador do ALBERT.** O artigo afirma ter usado "o tokenizador BERT padrão" (compatível com BERT-base-uncased). Para quatro dos cinco modelos isso coincide com o tokenizador próprio do checkpoint. O ALBERT, porém, usa vocabulário SentencePiece: alimentá-lo com IDs do vocabulário BERT não corresponderia às suas embeddings. Adotou-se o tokenizador próprio do ALBERT (carregado por `AutoTokenizer`), o que exige o pacote `sentencepiece` no ambiente.

### 7.2 Ajuste do código

O pipeline (Módulos A–D) e o protocolo de treino **não mudaram** — mesmos dados (amostragem, deduplicação, undersampling e split com `random_state=42`), protocolo "artigo" (10 épocas fixas, sem early stopping e sem seleção de checkpoint no teste) e mesmos hiperparâmetros para todos os modelos (lr 5e-5, batch 8, `weight_decay=0.01`, `max_length=128`). A mudança foi apenas na forma de **escolher o modelo e organizar as saídas**:

- **Notebook — nova célula de configuração:** dicionário `modelos_artigo` (apelido → checkpoint), variável `modelo_escolhido` como único ponto de troca entre experimentos, pastas derivadas (`pasta_modelo`, `pasta_checkpoints`, `pasta_resultados`) e a função `proxima_rodada`, que numera automaticamente as execuções do dia (maior rodada existente + 1), eliminando a renomeação manual de arquivos.
- **Notebook — células C e D:** o checkpoint passa a ser enviado a `carregar_tokenizer_e_modelo(checkpoint_modelo)`; o modelo vai para `modelo/<modelo>/` (checkpoints em `modelo/<modelo>/checkpoints/`) e os resultados para `resultados/<modelo>/resultado_final-<rodada>_<ddmmaaaa>.csv` e `curva_perda-<rodada>_<ddmmaaaa>.png`. O CSV ganhou as colunas de rastreabilidade `modelo`, `checkpoint` e `rodada`. A variável `modelo_tb` foi renomeada para `modelo_pre_treinado`.
- **Módulo D (`performance_assessment.gerar_curva_perda`):** novo parâmetro opcional `titulo`, para identificar o modelo no gráfico (padrão mantém o título anterior).
- **Procedimento de execução:** kernel reiniciado entre um modelo e outro (memória limpa e tempos mais comparáveis), com conferência no log dos números de controle do dataset (2.947 registros — 2.446 legítimos, 501 phishing; treino {1: 351, 0: 350}; teste {0: 151, 1: 150}).

**Validação do código novo.** O TinyBERT foi reexecutado com o código parametrizado (02/10, rodada 1) e reproduziu exatamente o resultado do lote de 01/10: Acurácia 0,993355 (1 FP + 1 FN), idêntica a 5 das 6 execuções anteriores, com curva de perda equivalente. Isso confirma que a parametrização não alterou o pipeline e valida, por extensão, as execuções dos demais modelos.

> **Registro de incidente operacional (01–02/10) — corrigido em 02/10.** Durante esta etapa, o notebook e o módulo `performance_assessment.py` voltaram, sem aviso, a versões anteriores, desfazendo alterações já aplicadas (o efeito visível foi um `TypeError` por parâmetro inexistente). A primeira hipótese registrada — abas antigas do Jupyter Lab salvando por cima — estava **errada**. A causa real foi o processo de gravação dos arquivos a partir da sessão do assistente (Claude): ao reaproveitar o mesmo caminho temporário de um envio anterior e enviar logo em seguida, o computador recebia a versão **anterior** daquele caminho (confirmado em 02/10: o arquivo gravado era idêntico, byte a byte, ao envio anterior, com horário de modificação igual ao da nova gravação). Correção adotada: caminho temporário único a cada envio e conferência de toda gravação por leitura do arquivo de volta (comparação de conteúdo). Notebook e módulo foram restaurados e conferidos. Lição para a reprodutibilidade: uma gravação "concluída sem erro" não prova que o conteúdo pretendido chegou ao destino — só a verificação direta do arquivo resultante prova (mesmo princípio da lição central do vazamento de dados, Seção 1.2), e cada estado validado deve ser versionado (commit no Git).

### 7.3 Tabela consolidada de resultados (piloto — 1 execução por modelo)

**Desempenho preditivo** (n = 301: 150 phishing + 151 legítimos)

| Modelo | Acc. | Prec. | Rec. | F1 | FP | FN | Erros |
|---|---|---|---|---|---|---|---|
| TinyBERT (PHILDER) | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 2 |
| MiniLM | 0,9867 | 0,9803 | 0,9933 | 0,9868 | 3 | 1 | 4 |
| DistilBERT | **0,9967** | 0,9934 | **1,0000** | **0,9967** | 1 | 0 | **1** |
| MobileBERT | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 2 | 1 | 3 |
| ALBERT | 0,9701 | 0,9732 | 0,9667 | 0,9699 | 4 | 5 | 9 |

**Tempos e eficiência computacional** (CPU local; tempo de treino bruto inclui a avaliação por época)

| Modelo | T. Treino bruto (s) | T. Aval. por época (s) | T. Treino líquido (s) | T. Teste (s) | Detecção (ms) | TES (bruto) | TES (líquido) | IES | RTDE |
|---|---|---|---|---|---|---|---|---|---|
| TinyBERT (PHILDER) | **382,19** | 21,52 | **360,67** | **2,15** | **7,14** | **0,002599** | **0,002754** | **0,4624** | **139,18** |
| MiniLM | 1.430,11 | 92,07 | 1.338,05 | 9,01 | 29,95 | 0,000690 | 0,000737 | 0,1095 | 32,95 |
| DistilBERT | 1.629,70 | 153,40 | 1.476,30 | 15,17 | 50,41 | 0,000612 | 0,000675 | 0,0657 | 19,77 |
| MobileBERT | 1.382,11 | 103,63 | 1.278,47 | 10,24 | 34,02 | 0,000716 | 0,000774 | 0,0967 | 29,10 |
| ALBERT | 2.929,51 | 362,99 | 2.566,52 | 35,91 | 119,30 | 0,000331 | 0,000378 | 0,0270 | 8,13 |

> Para o TinyBERT, a tabela usa a execução de validação de 02/10 (mesmas condições dos demais modelos). Referência do lote completo de 01/10 (6 execuções): Acc. 0,9928 ± 0,0014; T. Treino 406,40 ± 11,02 s; T. Teste 2,34 ± 0,20 s; TES 0,002444 ± 0,000063; IES 0,4258 ± 0,0330; RTDE 128,16 ± 9,92.

**Comparação com o artigo** (undersampling; Acc. e F1 da Tabela 4 — média ± DP de 3 execuções; demais valores da Tabela 3)

| Modelo | Acc. artigo | Acc. replicação | Rec. artigo | Rec. replicação | T. Treino artigo (s) | T. Treino replicação (s) | T. Teste artigo (s) | T. Teste replicação (s) |
|---|---|---|---|---|---|---|---|---|
| TinyBERT (PHILDER) | 0,9927 ± 0,0006 | 0,9934 | 0,9960 | 0,9933 | 180,52 | 382,19 | 6,22 | 2,15 |
| MiniLM | 0,9889 ± 0,0007 | 0,9867 | 0,9982 | 0,9933 | 103,52 | 1.430,11 | 6,21 | 9,01 |
| DistilBERT | 0,9934 ± 0,0005 | 0,9967 | 1,0000 | 1,0000 | 111,83 | 1.629,70 | 4,48 | 15,17 |
| MobileBERT | 0,9904 ± 0,0004 | 0,9900 | 0,9865 | 0,9933 | 219,45 | 1.382,11 | 9,76 | 10,24 |
| ALBERT | 0,9834 ± 0,0008 | 0,9701 | 0,9864 | 0,9667 | 236,81 | 2.929,51 | 9,09 | 35,91 |

**Custo relativo ao TinyBERT** (múltiplo do tempo do TinyBERT)

| Modelo | Treino — artigo | Treino — replicação | Teste — artigo | Teste — replicação |
|---|---|---|---|---|
| MiniLM | 0,57× | 3,74× | 1,00× | 4,19× |
| DistilBERT | 0,62× | 4,26× | 0,72× | 7,06× |
| MobileBERT | 1,22× | 3,62× | 1,57× | 4,76× |
| ALBERT | 1,31× | 7,67× | 1,46× | 16,70× |

### 7.4 Achados do piloto

1. **A ordenação de acurácia é idêntica à do artigo.** DistilBERT > TinyBERT > MobileBERT > MiniLM > ALBERT, tanto na Tabela 4 do artigo quanto nesta replicação — com dataset substituto, sem código dos autores e com hiperparâmetros não especificados no artigo. TinyBERT (0,9934 vs. 0,9927) e MobileBERT (0,9900 vs. 0,9904) ficam a menos de 1 e-mail do artigo; MiniLM a ~1 e-mail. O DistilBERT repete inclusive o recall 1,0000 do artigo. Ressalva: com n = 301, o erro-padrão da acurácia é ≈ ±0,5 p.p. (≈ 1,5 e-mail), de modo que diferenças de 1–2 e-mails entre modelos não são distinguíveis com uma execução.
2. **A tese do PHILDER se confirma com mais força nas medições próprias do que no próprio artigo.** O TinyBERT é 3,6–7,7× mais rápido no treino e 4,2–16,7× mais rápido na inferência que os demais, e a ordem de custo acompanha o tamanho das arquiteturas. A diferença de acurácia em relação ao melhor modelo (DistilBERT) é de 1 e-mail, ao custo de 4,3× no treino e 7,1× na inferência — exatamente o trade-off que TES, IES e RTDE foram desenhados para expor.
3. **A Tabela 3 do artigo não é compatível com o custo das arquiteturas.** No artigo, os tempos variam numa faixa estreita (≈ 0,6–1,3× no treino, 0,7–1,6× no teste, em relação ao TinyBERT) e a ordem se inverte em relação ao tamanho dos modelos (MiniLM e DistilBERT mais rápidos que o TinyBERT). **Hipótese (não confirmada):** medição em GPU, onde modelos pequenos não saturam o hardware e os tempos passam a ser dominados por custos fixos (transferência de dados, carregamento, sincronização), comprimindo e embaralhando as diferenças. Em CPU — cenário mais próximo de um dispositivo de borda, o alvo do PHILDER — o custo computacional domina e a vantagem do TinyBERT aparece integralmente. Se confirmada, a vantagem do PHILDER seria **maior** justamente no ambiente para o qual foi proposto.
4. **MobileBERT — instabilidade inicial.** A loss de treino começa em ≈ 1,7 × 10⁶ e cai a ≈ 0 em menos de 20 passos; o modelo se recupera (acurácia alinhada ao artigo). Comportamento conhecido da arquitetura (camadas "NoNorm", sem LayerNorm, gerando logits muito grandes com a camada de classificação recém-inicializada), contido pelo gradient clipping padrão do `Trainer`. Consequência prática: a curva de perda em escala linear fica ilegível para este modelo.
5. **ALBERT — platô inicial e menor convergência.** A loss permaneceu em ≈ 0,69 (= ln 2, equivalente a um classificador que chuta 50/50) nas primeiras ~4 épocas, só passando a aprender a partir da 5ª; ao final das 10 épocas a loss de avaliação (≈ 0,17) ainda era bem maior que a dos demais (≈ 0,03–0,09). Isso explica a acurácia mais baixa e o maior afastamento em relação ao artigo (0,9701 vs. 0,9834). O ALBERT é conhecido por exigir learning rate menor e warmup no fine-tuning; **os hiperparâmetros foram deliberadamente mantidos iguais para todos os modelos** — ajustá-los por modelo a partir do resultado no teste reintroduziria o viés de seleção eliminado em 01/10 (Seção 3.4). Registrado como limitação.
6. **Custo de avaliação por época** varia de 5,6% (TinyBERT) a 12,4% (ALBERT) do tempo de treino bruto; o TES líquido não altera a ordenação dos modelos.

### 7.5 Viabilidade e próximos passos

- **Tempo de uma rodada completa (5 modelos):** ≈ 2h10 de treino (≈ 2,5 h com carregamento e teste). Como o TinyBERT já tem 6 execuções, completar os outros quatro modelos custa ≈ 2h20 por rodada adicional: ≈ 4h40 para chegar a 3 execuções por modelo (como o artigo) e ≈ 11h40 para 6.
- **Pendente de decisão:** número de execuções por modelo na comparação final (3 ou 6).
- Avaliar escala logarítmica (ou recorte dos passos iniciais) em `gerar_curva_perda`, para tornar legíveis curvas como a do MobileBERT.
- Análise de erros: inspecionar os e-mails classificados incorretamente (em especial o phishing perdido por TinyBERT, MiniLM e MobileBERT e acertado pelo DistilBERT).
- Os tempos dependem do estado da máquina (ex.: TinyBERT 382 s em 02/10 vs. 406 ± 11 s em 01/10, mesmo código): as rodadas finais devem ser executadas em condições controladas e equivalentes para todos os modelos.

---

## 8. Reprodutibilidade entre ambientes (Google Colab — 03/10/2026)

### 8.1 Contexto

Para permitir o trabalho remoto (sem o computador local), o mesmo notebook e o mesmo pacote `philder/` foram executados no **Google Colab**, a partir de uma cópia do projeto no Google Drive. O objetivo foi verificar se o pipeline produz os mesmos resultados num ambiente diferente — outro sistema operacional, outro processador e outras versões de bibliotecas — com o mesmo código, os mesmos dados de origem e as mesmas sementes.

| | PC local | Google Colab |
|---|---|---|
| Sistema operacional | Windows | Linux |
| Processador | CPU do notebook local | 2 núcleos virtuais AMD EPYC 7B12 (compartilhado), sem GPU |
| Python | 3.12.13 | 3.13.15 |
| torch | 2.14.0+cpu | 2.11.0+cpu |
| transformers | 5.16.1 | 5.17.0 |

### 8.2 Falha de reprodutibilidade identificada: ordem de leitura dos arquivos

**Sintoma.** A primeira execução do TinyBERT no Colab obteve Acurácia 0,9767 (**7 erros**: 5 FP + 2 FN), contra 0,9934 (2 erros: 1 FP + 1 FN) no PC local — um salto grande demais para ser explicado por ruído numérico entre máquinas. As contagens de controle no log eram **idênticas** nos dois ambientes (2.947 registros — 2.446 legítimos, 501 phishing; treino {1: 351, 0: 350}; teste {0: 151, 1: 150}) e não indicavam nenhuma diferença.

**Diagnóstico.** Foi calculada uma impressão digital (hash MD5) do **conteúdo** dos conjuntos — os textos de `body_processado`, ordenados e concatenados — nos dois ambientes, após o Módulo C:

| Conjunto | PC local | Colab (antes da correção) |
|---|---|---|
| Balanceado | `c818699ea3118932f9a64f9dcc16169e` | `b10ef31410509e5ab6ec325452f6f811` |
| Treino | `c7f1187377e71ac94aa2f100ee2d3e08` | `f9c9cc874ed13f19ac3ad186fa4a1ced` |
| Teste | `243d7b22075cf4f6d88ecc63c04005ed` | `b6b8d650b77e7c78ddac5da4c945b6f9` |

Os três conjuntos diferiam. A causa estava em `importar_corpus_spamassassin` (Módulo A), que lê os e-mails com `glob.glob(...)` **sem ordenação**: no Windows (NTFS) a lista vem em ordem alfabética (`00001…`, `00002…`, `00003…`); no Linux do Colab, em ordem arbitrária (`02159…`, `02058…`, `00737…`). Essa ordem se propaga pelo pipeline: (1) a deduplicação (`keep="first"`) escolhe outro representante entre duplicatas; (2) o DataFrame consolidado fica em outra ordem e o undersampling, com a mesma semente, sorteia **outros 501 legítimos**; (3) o split 70/30, com a mesma semente, gera **outro conjunto de teste**. O resultado de 0,9767 foi, portanto, medido sobre **outro dataset** e foi descartado.

**Correção.** `arquivos_ham = sorted(glob.glob(os.path.join(pasta_extraida, "*")))`. Como o Windows já devolvia os arquivos em ordem alfabética, a ordenação reproduz exatamente a ordem sempre usada no PC local — **nenhum resultado local anterior é afetado** — e torna a leitura determinística em qualquer sistema. Após a correção, os hashes do Colab passaram a coincidir com os do PC local. *Situação em 03/10:* correção aplicada na cópia do Colab e, no mesmo dia, incorporada ao código do PC local (`philder/database_consolidation.py`, com comentário datado e a linha anterior preservada como comentário). Pendentes: commit/push para o repositório GitHub e verificação, no PC local, de que os três hashes permanecem iguais.

**Lição para a reprodutibilidade.** Sementes fixas não garantem reprodutibilidade se a **ordem de entrada dos dados** depender do sistema operacional. Contagens idênticas não provam conteúdo idêntico — só a verificação direta do conteúdo (impressão digital) revelou a diferença. É o mesmo princípio das lições anteriores do projeto (vazamento de dados, Seção 1.2; verificação de gravação, Seção 7.2).

### 8.3 Resultados no Colab após a correção (TinyBERT, 6 execuções)

| Execução | Acc. | Prec. | Rec. | F1 | FP | FN | T. Treino bruto (s) | T. Aval. por época (s) | T. Teste (s) | Detecção (ms) | TES | IES | RTDE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 630,78 | 62,55 | 6,54 | 21,74 | 0,001575 | 0,1518 | 45,70 |
| 2 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 665,95 | 65,02 | 6,05 | 20,11 | 0,001492 | 0,1641 | 49,41 |
| 3 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 640,24 | 63,59 | 6,69 | 22,22 | 0,001552 | 0,1485 | 44,70 |
| 4 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 640,13 | 62,56 | 6,03 | 20,04 | 0,001552 | 0,1646 | 49,56 |
| 5 | 0,9900 | 0,9933 | 0,9867 | 0,9900 | 1 | 2 | 628,57 | 61,18 | 5,64 | 18,74 | 0,001575 | 0,1755 | 52,84 |
| 6 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 630,86 | 61,50 | 6,23 | 20,70 | 0,001575 | 0,1594 | 47,98 |
| **Média ± DP** | **0,9928 ± 0,0014** | **0,9933 ± 0,0000** | **0,9922 ± 0,0027** | **0,9928 ± 0,0014** | — | — | **639,42 ± 13,93** | **62,73 ± 1,41** | **6,20 ± 0,38** | **20,59 ± 1,26** | **0,001553 ± 0,000032** | **0,1607 ± 0,0098** | **48,36 ± 2,94** |

T. Treino líquido: 576,69 ± 12,65 s; TES (líquido): 0,001722 ± 0,000036.

### 8.4 Comparação entre ambientes

| | Colab, 03/10 (n=6) | PC local, 01/10 (n=6) | Artigo — TinyBERT (2) |
|---|---|---|---|
| Acurácia | **0,9928 ± 0,0014** | **0,9928 ± 0,0014** | 0,9927 ± 0,0006 |
| Precisão | 0,9933 ± 0,0000 | 0,9922 ± 0,0027 | 0,9896 |
| Recall | 0,9922 ± 0,0027 | 0,9933 ± 0,0000 | 0,9960 |
| F1 | 0,9928 ± 0,0014 | 0,9928 ± 0,0013 | 0,9927 ± 0,0006 |
| Execuções com 2 erros (1 FP + 1 FN) | 5 de 6 | 5 de 6 | — |
| Execução com 3 erros | 1 (+1 FN) | 1 (+1 FP) | — |
| T. Treino bruto (s) | 639,42 ± 13,93 | 406,40 ± 11,02 | 180,52 |
| Avaliação por época (% do treino) | 9,8% | 5,6% | — |
| T. Teste (s) | 6,20 ± 0,38 | 2,34 ± 0,20 | 6,22 |
| Detecção (ms) | 20,59 ± 1,26 | 7,79 ± 0,66 | 12,98 |
| TES | 0,001553 ± 0,000032 | 0,002444 ± 0,000063 | 0,0055 |
| IES | 0,1607 ± 0,0098 | 0,4258 ± 0,0330 | 0,1596 |
| RTDE | 48,36 ± 2,94 | 128,16 ± 9,92 | 76,47 |

### 8.5 Leitura

1. **O desempenho preditivo se reproduz entre ambientes.** Com outro sistema operacional, outro processador e outras versões de Python, torch e transformers, Acurácia e F1 têm média e desvio-padrão iguais até a 4ª casa decimal, e o mesmo padrão de erros se repete (5 de 6 execuções com exatamente 1 FP + 1 FN). A única diferença é a execução fora do padrão (um FN a mais no Colab, um FP a mais no PC local) — variação do tamanho de um e-mail na fronteira de decisão. Isso reforça a confiabilidade do resultado da replicação.
2. **A diferença inicial vinha inteiramente dos dados, não do ambiente.** Antes da correção da ordem de leitura, o mesmo código no Colab produziu 7 erros; depois dela, 2.
3. **Os tempos escalam com o hardware, e de forma desigual entre as métricas.** No Colab (2 núcleos virtuais compartilhados), o treino foi ≈ 1,6× mais lento e o teste ≈ 2,6× mais lento que no PC local; a inferência em lote perde mais com poucos núcleos. A avaliação por época também pesa mais (9,8% vs. 5,6% do treino). Consequência: **as métricas de eficiência não são transportáveis entre ambientes por proporção simples** — TES, IES e RTDE só são comparáveis quando medidos no mesmo hardware, e o hardware precisa ser reportado junto com elas.
4. **Coincidência a ser lida com cautela.** O tempo de teste e o IES do Colab (6,20 s; 0,161) ficaram quase iguais aos do artigo (6,22 s; 0,160), enquanto o treino do artigo (180,52 s) é ≈ 3,5× mais rápido que o do Colab. Esses valores não se encaixam num único hardware conhecido, o que reforça que a Tabela 3 do artigo não admite explicação simples (Seção 7.4); nada se pode concluir sobre o hardware dos autores a partir disso.

### 8.6 Pendências

- ~~Incorporar `sorted(glob.glob(...))` ao código do PC local~~ (feito em 03/10). Falta: commit/push para o GitHub e verificar no PC local que os três hashes permanecem iguais (`c818699e…`, `c7f11873…`, `243d7b22…`).
- Registrar, em toda execução fora do PC local, o hardware e as versões de Python/torch/transformers junto com os resultados (resultados do Colab mantidos separados dos do PC local).
- Avaliar fixar as versões das bibliotecas (`requirements.txt`) no pacote de reprodutibilidade.

### 8.7 Expansão no Colab: os 5 modelos (1 execução por modelo, 03–04/10/2026)

Mesmo ambiente da Seção 8.1 (Colab, CPU com 2 núcleos virtuais, sem GPU), mesmo código (com a ordenação da Seção 8.2), dados e protocolo. Uma execução por modelo (TinyBERT: execução 6 de 03/10; MiniLM: execução 2 de 03/10; demais: 04/10).

**Métricas preditivas — Colab vs. PC local (piloto de 02/10, Seção 7.3)**

| Modelo | Acc. Colab | Erros Colab (FP + FN) | Acc. PC local | Erros PC local (FP + FN) | Acc. artigo (Tab. 4) |
|---|---|---|---|---|---|
| TinyBERT (PHILDER) | 0,9934 | 2 (1 + 1) | 0,9934 | 2 (1 + 1) | 0,9927 ± 0,0006 |
| MiniLM | 0,9867 | 4 (2 + 2) | 0,9867 | 4 (3 + 1) | 0,9889 ± 0,0007 |
| DistilBERT | **0,9967** | 1 (1 + 0) | **0,9967** | 1 (1 + 0) | 0,9934 ± 0,0005 |
| MobileBERT | 0,9900 | 3 (1 + 2) | 0,9900 | 3 (2 + 1) | 0,9904 ± 0,0004 |
| ALBERT | **0,9900** | **3 (1 + 2)** | **0,9701** | **9 (4 + 5)** | 0,9834 ± 0,0008 |

Precisão / Recall / F1 no Colab: TinyBERT 0,9933 / 0,9933 / 0,9933; MiniLM 0,9867 / 0,9867 / 0,9867; DistilBERT 0,9934 / 1,0000 / 0,9967; MobileBERT 0,9933 / 0,9867 / 0,9900; ALBERT 0,9933 / 0,9867 / 0,9900.

**Tempos e eficiência computacional no Colab**

| Modelo | T. Treino bruto (s) | Aval. por época (% do treino) | T. Teste (s) | Detecção (ms) | TES | IES | RTDE | Treino × TinyBERT | Teste × TinyBERT |
|---|---|---|---|---|---|---|---|---|---|
| TinyBERT (PHILDER) | **630,86** | 9,7% | **6,23** | **20,70** | **0,001575** | **0,1594** | **47,98** | 1,0× | 1,0× |
| MiniLM | 2.244,40 | 10,4% | 23,16 | 76,94 | 0,000440 | 0,0426 | 12,82 | 3,6× | 3,7× |
| DistilBERT | 5.718,36 | 11,3% | 63,67 | 211,53 | 0,000174 | 0,0157 | 4,71 | 9,1× | 10,2× |
| MobileBERT | 3.716,17 | 11,2% | 39,67 | 131,79 | 0,000266 | 0,0250 | 7,51 | 5,9× | 6,4× |
| ALBERT | 11.391,16 (≈ 3h10) | 12,6% | 144,86 | 481,26 | 0,000087 | 0,0068 | 2,06 | 18,1× | 23,3× |

**Leitura.** Quatro dos cinco modelos repetem exatamente a acurácia e o número de erros do PC local (DistilBERT e TinyBERT com os mesmos tipos de erro; MiniLM e MobileBERT com troca de um FP por um FN — ruído de fronteira). É mais uma confirmação de que os dados são idênticos nos dois ambientes (Seção 8.2). A ordem de acurácia no Colab — DistilBERT > TinyBERT > MobileBERT = ALBERT > MiniLM — difere da do artigo e do PC local apenas na posição do ALBERT (Seção 8.8). Viabilidade: uma rodada completa dos 5 modelos levou ≈ 6,5 h no Colab (ALBERT sozinho > 3 h), o que torna arriscado rodar lotes completos na versão gratuita (sessão máxima e desconexão por inatividade).

### 8.8 Instabilidade de fine-tuning do ALBERT

Com os mesmos dados, código e hiperparâmetros, o ALBERT obteve 9 erros no PC local e 3 no Colab — a única diferença relevante entre os ambientes em todo o experimento. As curvas de perda explicam:

| | ALBERT — PC local (02/10) | ALBERT — Colab (04/10) |
|---|---|---|
| Primeiras épocas | loss de treino parada em ≈ 0,69 (= ln 2, equivalente a classificar ao acaso) por ~4 épocas; loss de avaliação **subindo** até ≈ 0,71 | loss de treino cai de ≈ 0,6 para ≈ 0,05 já na 1ª época; loss de avaliação começa em ≈ 0,14 |
| Início efetivo do aprendizado | 5ª época | 1ª época |
| Loss de avaliação final | ≈ 0,17 | ≈ 0,08 |
| Resultado | 0,9701 (9 erros) | 0,9900 (3 erros) |

Mesmo no Colab, a loss de treino apresentou picos fortes ao longo do treino (≈ 0,56 perto do passo 130; ≈ 0,33 perto do passo 440), bem maiores que os dos demais modelos.

**Interpretação.** O fine-tuning do ALBERT com os hiperparâmetros adotados (lr 5e-5, sem warmup) comporta-se de forma quase binária: o modelo fica ou não preso num platô inicial, e pequenas diferenças numéricas entre ambientes (processador, versões de bibliotecas) decidem o desfecho — com efeito de ≈ 6 e-mails (≈ 2 p.p. de acurácia) no resultado final. Consequências:

1. **Uma execução não representa o ALBERT.** O resultado tende a ser bimodal; nas rodadas múltiplas, deve-se registrar quantas execuções ficaram "presas" no platô, além da média (a média de dois regimes distintos esconde o fenômeno).
2. **Lacuna do artigo.** O artigo reporta 0,9834 ± 0,0008 para o ALBERT. Uma variação tão pequena não é compatível com a instabilidade observada: ou as três execuções dos autores coincidiram no mesmo regime, ou usaram algum recurso de estabilização não descrito (warmup, learning rate menor), ou as execuções não variaram entre si (mesma ressalva já feita ao ± do artigo, Seção 3.4).
3. **Os hiperparâmetros continuam iguais para todos os modelos.** "Corrigir" o ALBERT ajustando hiperparâmetros a partir do resultado no teste reintroduziria o viés de seleção eliminado em 01/10 (Seção 3.4). A instabilidade é reportada como característica do modelo nessas condições.

### 8.9 Efeito do hardware na vantagem do TinyBERT

Lentidão do Colab (2 núcleos virtuais compartilhados) em relação ao PC local, por modelo:

| Modelo | Dimensão interna | Treino (Colab ÷ PC) | Teste (Colab ÷ PC) |
|---|---|---|---|
| TinyBERT | 312 | 1,65× | 2,90× |
| MiniLM | 384 | 1,57× | 2,57× |
| MobileBERT | 128 (gargalo) / 512 | 2,69× | 3,87× |
| DistilBERT | 768 | 3,51× | 4,20× |
| ALBERT | 768 | 3,89× | 4,03× |

Os modelos com dimensão interna 768 foram os mais penalizados pela máquina mais fraca. **Hipótese (não verificada):** as matrizes dos modelos menores cabem melhor na memória cache do processador, enquanto as dos maiores dependem mais da memória principal — gargalo típico de uma máquina virtual compartilhada.

Consequência: a vantagem de tempo do TinyBERT sobre os demais modelos passa de 3,6–7,7× no treino e 4,2–16,7× na inferência (PC local, Seção 7.4) para **3,6–18,1× no treino e 3,7–23,3× na inferência** (Colab). **Quanto mais limitado o hardware, maior a vantagem do PHILDER** — evidência, em dois ambientes de CPU, a favor da tese do artigo para dispositivos com recursos limitados, e coerente com a hipótese levantada para a Tabela 3 (Seção 7.4): em hardware potente (GPU), as diferenças entre os modelos tenderiam a encolher.

### 8.10 Pendências adicionais

- Nas rodadas múltiplas finais, registrar para o ALBERT (e para os demais modelos) se houve platô inicial, a partir da curva de perda de cada execução.
- Avaliar, como experimento complementar e explicitamente separado do protocolo principal, o efeito de warmup/learning rate menor sobre a estabilidade do ALBERT — sem usar o resultado no teste para escolher a configuração do protocolo principal.
- Experimento GPU vs. CPU (Colab/Kaggle com GPU) para testar a hipótese de compressão das diferenças de tempo em hardware potente.
