# Análise dos Resultados — Replicação PHILDER
### Aplicações Distribuídas — PPGEE/UnB | Fases 2 e 3

> Este documento registra, em ordem cronológica, cada execução do pipeline de replicação do PHILDER (BISPO, G. D. et al., IEEE Access, 2025) e compara os resultados obtidos com os do artigo original. Para a fundamentação teórica das métricas, a análise crítica da normalização de TES/RTDE na Tabela 4 do artigo e o detalhamento estatístico completo do achado de vazamento de dados (Seções 3.5, 5.5 e 5.6), ver `Embasamento_Teorico_Metricas_PHILDER.md`.

> **Reorganização de 08/10/2026.** A conclusão foi movida para o final do documento (Seção 9) e as antigas Seções 7, 8 e 9 passaram a ser 6, 7 e 8 (referências cruzadas atualizadas). O resultado de referência oficial passou a ser o **lote final de 07/10/2026 — 5 modelos × 6 execuções, undersampling 502/501** (Seções 5 e 8). Os registros anteriores foram preservados; quando um trecho ficou superado, isso é indicado no próprio trecho.

**Estrutura do documento**

| Seção | Conteúdo |
|---|---|
| 1 | Linha do tempo de todas as ações e medições; notas sobre o tamanho do teste e sobre as trocas de resultado de referência |
| 2 | Resultados do artigo original (valores de referência e inconsistências internas) |
| 3 | TinyBERT (PHILDER) por execução — Fase 2 (17/09–25/09) e protocolo fiel ao artigo (01/10) |
| 4 | Análise evolutiva — o que mudou e por quê, de 17/09 a 07/10 |
| 5 | Comparação final: resultado de referência atual vs. artigo |
| 6 | Expansão para os 5 modelos — piloto de 02/10 |
| 7 | Reprodutibilidade entre ambientes — Google Colab (03–04/10) |
| 8 | Automação, alinhamento do undersampling e lote final (04–07/10) |
| 9 | Conclusão, lições metodológicas e pendências consolidadas |

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
| 01/10/2026 | Mesmos dados de 25/09(b); **protocolo de treino fiel ao artigo** (`treinar_modelo(..., protocolo="artigo")`): 10 épocas fixas, sem early stopping e sem seleção de checkpoint pelo F1 do teste; avaliação por época mantida só para a curva de perda; sem gravação de checkpoints; tempo de avaliação por época registrado à parte (`tempo_treino_liquido`) | Remove o **viés de seleção de modelo no conjunto de teste** (o lote 25/09(b) usava o teste para decidir a parada e o checkpoint final). Seis execuções repetidas (arquivos `resultado_final-[1-6]_01102026.csv`, curvas `curva_perda-[1-6]_01102026.png`). Ver Seção 3.4. No mesmo dia: **correção da tabela de referência do artigo** (os tempos vinham da linha DistilBERT (2) da Tabela 3; ver Seção 2); `weight_decay=0.01` e split estratificado mantidos por decisão explícita |
| 02/10/2026 | Mesmos dados e protocolo "artigo" de 01/10; **modelo passado como parâmetro** e padrão de saída por modelo (`modelo/<modelo>/`, `resultados/<modelo>/`) | **Expansão para os 5 modelos do artigo** (TinyBERT, MiniLM, DistilBERT, MobileBERT, ALBERT) — rodada-piloto, 1 execução por modelo, para avaliar viabilidade, tempos e resultados. TinyBERT reexecutado com o código novo como validação. Registrado também um incidente operacional de gravação de arquivos (Seção 6.2). Ver Seção 6 |
| 03/10/2026 | Mesmo código, dados e protocolo; execução no **Google Colab** (CPU, 2 núcleos virtuais AMD EPYC 7B12, sem GPU; Linux) | **Reprodutibilidade entre ambientes.** Descoberta e correção de dependência da ordem de leitura dos arquivos do SpamAssassin (`glob` sem ordenação: alfabética no Windows, arbitrária no Linux), que gerava outro dataset com as mesmas sementes. Após a correção (`sorted(glob.glob(...))`, aplicada no Colab e, no mesmo dia, no código do PC local), seis execuções do TinyBERT reproduzem as métricas preditivas do PC local. Hashes do PC local conferidos após a correção (inalterados). Ver Seção 7 |
| 03–04/10/2026 | Google Colab (mesmo ambiente da Seção 7), 1 execução de cada um dos 5 modelos | **Expansão no Colab**: 4 dos 5 modelos repetem exatamente a acurácia do piloto local; o ALBERT passa de 9 para 3 erros — evidência de **instabilidade de fine-tuning** (platô inicial em ln 2 no PC, ausente no Colab). Na CPU de 2 núcleos, a vantagem de tempo do TinyBERT cresce (até 18× no treino e 23× na inferência). Ver Seções 7.7–7.9 |
| 04/10/2026 | Mesmo código de treino; novo script **`rodar_lote.py`** (raiz do projeto) | **Automação da execução em lote**: um processo Python separado por treinamento (equivale a reiniciar o kernel entre execuções), execução sequencial (sem paralelismo, para não contaminar os tempos), numeração automática das rodadas e registro automático de ambiente, versões e impressões digitais dos dados em cada CSV. O notebook foi mantido sem alterações. Ver Seção 8.2 |
| 07/10/2026 (manhã) | Dados de 25/09(b) (501/501), protocolo "artigo", TinyBERT, **6 execuções via `rodar_lote.py`** (rodadas 2–7; rodada 1 do dia feita pelo notebook) | **Validação do script**: Acc. 0,9928 ± 0,0025 (mesma média do lote de 01/10) e hashes idênticos aos de referência 501/501. Arquivos movidos depois para `resultados/tinybert/teste e validacao/`. Ver Seção 8.3 |
| 07/10/2026 | **Undersampling alinhado ao artigo: 502 legítimos + 501 phishing = 1.003** (Figura 2 do artigo) | `aplicar_undersampling(..., n_por_classe=...)` (Módulo C), variável `undersampling_alvo` no notebook (células de configuração, C e D) e opção `--undersampling` no `rodar_lote.py` (padrão "artigo"). Validação com 1 execução do TinyBERT pelo notebook (Acc. 0,9900; 3 FP, 0 FN). O conjunto de teste muda (novo sorteio do split). Ver Seção 8.4 |
| 07/10/2026 (09h06–23h39) | 502/501, protocolo "artigo", **5 modelos × 6 execuções** via `rodar_lote.py --modelos todos --rodadas 6`, PC local | **Lote final — resultado de referência oficial atual.** 30 execuções sem falha, mesmos dados em todas (hashes idênticos). TinyBERT 0,9934 ± 0,0021; DistilBERT 0,9928 ± 0,0014; MobileBERT 0,9856 ± 0,0034; ALBERT 0,9839 ± 0,0044; MiniLM 0,9806 ± 0,0039. Ver Seções 5 e 8.5–8.10 |
| 08/10/2026 | — | Reorganização deste documento: linha do tempo completada, nota de estratificação corrigida (Seção 1.1), histórico de referências (Seção 1.3), comparação final atualizada (Seção 5), conclusão movida para o final (Seção 9) |

### 1.1 Nota sobre "n (teste)": não é uma contagem só de phishing

Em todas as tabelas deste documento e do Embasamento Teórico, **"n (teste)" é o tamanho total do conjunto de teste — as duas classes somadas, não uma contagem isolada de e-mails de phishing.** O artigo descreve a divisão como *"data were then split into 70% training and 30% testing sets using the datasets library from Hugging Face"*, e essa proporção é aplicada sobre o dataset **já balanceado pelo undersampling** (Módulo C), não sobre o dataset bruto/desbalanceado:

$$501\ \text{phishing} + 501\ \text{legítimos} = 1.002\ \text{registros} \quad\Rightarrow\quad 1.002 \times 0{,}30 = 300{,}6 \approx 301$$

Ou seja, os "301" (ou "289", na execução de 22/09 com deduplicação pós-consolidação) que aparecem nas tabelas abaixo são o **total** do conjunto de teste, não 301 e-mails de phishing. O split é feito por `dividir_treino_teste` (Módulo C) com `train_test_split(..., test_size=0.30, stratify=df[coluna_rotulo], random_state=42)`, ou seja, **estratificado por classe** — opção mantida por decisão explícita em 01/10 (o artigo usa a biblioteca `datasets` do Hugging Face e não informa se estratifica). Com 1.002 registros 50/50, a estratificação produz teste com 151 legítimos + 150 phishing = 301, o que bate com a matriz de confusão reconstruída de 21/09 (ver Seção 5.5 do Embasamento Teórico) e com a checagem direta do lote 25/09(b) (`Teste: {0: 151, 1: 150}`).

**A partir de 07/10 (undersampling do artigo, Seção 8.4):** 502 legítimos + 501 phishing = 1.003 registros ⇒ 1.003 × 0,30 = 300,9 ≈ 301 no teste. O tamanho do teste não muda (301: 151 legítimos + 150 phishing) e o treino passa a 702 registros (351 + 351). O **conteúdo** do teste, porém, muda — uma amostra a mais altera todo o sorteio —, por isso os resultados de 07/10 em diante não são diretamente comparáveis, e-mail a e-mail, com os anteriores.

> **Correção 08/10/2026:** a versão anterior desta nota afirmava que o split "não é estratificado por classe". O código (`stratify=df[coluna_rotulo]`) mostra o contrário; texto corrigido.

### 1.2 Nota sobre a mudança de resultado de referência: de 23/09 para 25/09

As execuções de 21/09 a 23/09 documentam corretamente o **processo de detecção** do vazamento de dados (precisão suspeita → checagem de duplicidade → correção desenhada). Porém, em 24/09 foi descoberto que a célula do notebook responsável pelo Módulo A nunca havia sido atualizada para de fato chamar `amostrar_quantidade_artigo` — ela mantinha um trecho de código residual de 19/09 que amostrava o phishing manualmente, sem qualquer deduplicação, e alimentava a consolidação do dataset diretamente com as listas brutas. Isso significa que a "correção" descrita para 23/09 nunca chegou a ser executada como tal; o pipeline que de fato gerou aquele resultado não pôde ser confirmado com certeza.

Corrigido esse bug, o dia 25/09 passou a rodar, pela primeira vez, o pipeline com a deduplicação de fato em execução — mas em **duas configurações distintas**, testadas em sequência no mesmo dia:

- **25/09(a) — dedup apenas do phishing** (`resultado_final_25092026_estavel[1-6].csv`): o Bug 1 (código morto do Módulo A) já estava corrigido, então `amostrar_quantidade_artigo(..., deduplicar_antes=True)` passou a rodar de verdade — mas, nesse ponto, a deduplicação prévia ainda só cobria o pool de phishing (Nazario); o pool de legítimos (SpamAssassin) continuava sendo amostrado sem deduplicação prévia (2.501 legítimos brutos).
- **25/09(b) — dedup de phishing e ham** (`resultado_final_25092026-3_estavel[1-6].csv`): extensão da mesma lógica também ao lado dos legítimos. Essa edição introduziu o Bug 2 (troca de variáveis, rótulos invertidos) — corrigido no mesmo dia — e o lote final, gerado já com o bug corrigido, é o primeiro cuja cadeia de execução foi verificada linha a código e confirmada por uma checagem programática explícita de sobreposição treino/teste (resultado: zero sobreposição, zero duplicatas internas).

Por isso, a partir de 25/09, **25/09(b) substituiu 23/09 como resultado de referência oficial da replicação** (posteriormente substituído — ver Seção 1.3), e 25/09(a) fica registrado como um resultado intermediário válido (também sem vazamento comprovado no phishing, mas com o pool de legítimos ainda não deduplicado) — útil para medir o efeito isolado de estender a deduplicação ao SpamAssassin (Seção 4). Os detalhes completos dos dois bugs, do processo de correção e da validação em três camadas estão na Seção 5.6 do Embasamento Teórico.

### 1.3 Histórico do resultado de referência

| Período | Resultado de referência | Por que foi substituído |
|---|---|---|
| 23/09–24/09 | 23/09 — dedup "desenhada" antes da consolidação (provisória) | O pipeline que gerou o resultado não pôde ser confirmado: a célula do Módulo A não executava a lógica de deduplicação (Seção 1.2) |
| 25/09–30/09 | 25/09(b) — TinyBERT, dedup de phishing e ham, 6 execuções, sobreposição treino/teste = 0 | Usava o conjunto de teste para seleção de modelo (early stopping e checkpoint de melhor F1), inflando as métricas em ≈ 0,4 p.p. (Seção 3.4) |
| 01/10–06/10 | Lote de 01/10 — TinyBERT, protocolo fiel ao artigo, 501/501, 6 execuções | Usado como referência de trabalho (a formalização ficou pendente na época). Superado por cobrir só o TinyBERT e por usar 1.002 amostras em vez das 1.003 do artigo |
| **07/10 em diante** | **Lote final de 07/10 — 5 modelos × 6 execuções, protocolo fiel ao artigo, undersampling 502/501, `rodar_lote.py`, PC local** | — (referência atual; Seções 5 e 8) |

Os resultados substituídos continuam registrados nas Seções 3, 6 e 7: cada um documenta uma etapa da correção do método e serve de comparação para isolar o efeito de cada mudança.

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

### Tabela 3 — Os cinco modelos no artigo (execução com undersampling)

| Modelo | Acurácia (Tab. 4, média ± DP de 3 exec.) | Recall (Tab. 3) | T. Treino (s) (Tab. 3) | T. Teste (s) (Tab. 3) |
|---|---|---|---|---|
| TinyBERT (PHILDER) | 0,9927 ± 0,0006 | 0,9960 | 180,52 | 6,22 |
| MiniLM | 0,9889 ± 0,0007 | 0,9982 | 103,52 | 6,21 |
| DistilBERT | 0,9934 ± 0,0005 | 1,0000 | 111,83 | 4,48 |
| MobileBERT | 0,9904 ± 0,0004 | 0,9865 | 219,45 | 9,76 |
| ALBERT | 0,9834 ± 0,0008 | 0,9864 | 236,81 | 9,09 |

> Pela própria Tabela 3, MiniLM e DistilBERT treinariam **mais rápido** que o TinyBERT (TES bruto 0,0096 e 0,0089 contra 0,0055), o que contradiz o texto, a Figura 3 e o tamanho das arquiteturas. Essa inconsistência motivou a medição dos cinco modelos no mesmo ambiente (Seções 6 e 8).

---

## 3. TinyBERT (PHILDER) por execução — Fase 2 e protocolo fiel ao artigo (17/09–01/10)

> Esta seção registra as execuções do TinyBERT até 01/10, todas com 501 phishing e (a partir de 19/09) undersampling 501/501. As execuções posteriores do TinyBERT (02/10, Colab, `rodar_lote.py` e lote final 502/501) estão nas Seções 6–8 e foram acrescentadas à tabela consolidada da Seção 3.3.

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

> Nota de arquivo: a execução 6 foi gravada com o nome padrão (`resultado_final.csv` / `curva_perda.png`). Renomeação concluída: `resultado_final-6_01102026.csv` e `curva_perda-6_01102026.png` estão em `resultados/` (conferido em 07/10).

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
| 25/09(b) | + dedup também dos legítimos, pipeline verificado (referência de 25/09 a 30/09; média de 6 exec.) | 0,9967 ± 0,0021 | 0,9978 ± 0,0034 | 0,9956 ± 0,0034 | 0,9967 ± 0,0021 | 352,24 ± 44,90 | 3,92 ± 0,29 | 13,03 ± 0,96 | 0,002872 ± 0,000406 | 0,255279 ± 0,018430 | 76,8516 ± 5,5475 |
| 01/10 | mesmos dados de 25/09(b), **protocolo fiel ao artigo** (10 ép. fixas, sem seleção no teste) (média de 6 exec.) | 0,9928 ± 0,0014 | 0,9922 ± 0,0027 | 0,9933 ± 0,0000 | 0,9928 ± 0,0013 | 406,40 ± 11,02 | 2,34 ± 0,20 | 7,79 ± 0,66 | 0,002444 ± 0,000063 | 0,4258 ± 0,0330 | 128,16 ± 9,92 |
| 02/10 | código parametrizado por modelo (piloto, Seção 6), 501/501, PC local (1 exec.) | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 382,19 | 2,15 | 7,14 | 0,002599 | 0,4624 | 139,18 |
| 03/10 | Google Colab (2 vCPU), após `sorted(glob)`, 501/501 (média de 6 exec., Seção 7.3) | 0,9928 ± 0,0014 | 0,9933 ± 0,0000 | 0,9922 ± 0,0027 | 0,9928 ± 0,0014 | 639,42 ± 13,93 | 6,20 ± 0,38 | 20,59 ± 1,26 | 0,001553 ± 0,000032 | 0,1607 ± 0,0098 | 48,36 ± 2,94 |
| 07/10 (manhã) | `rodar_lote.py`, 501/501, PC local (validação do script; média de 6 exec., Seção 8.3) | 0,9928 ± 0,0025 | 0,9933 ± 0,0042 | 0,9922 ± 0,0027 | 0,9928 ± 0,0025 | 444,37 ± 8,52 | 2,35 ± 0,06 | 7,80 ± 0,20 | 0,002235 ± 0,000045 | 0,4229 ± 0,0109 | 127,29 ± 3,29 |
| 07/10 | notebook, **undersampling 502/501** (validação; 1 exec., Seção 8.4) | 0,9900 | 0,9804 | 1,0000 | 0,9901 | 424,11 | 2,35 | 7,82 | 0,002334 | 0,4205 | 126,56 |
| 07/10 | **lote final**, `rodar_lote.py`, 502/501, PC local (**referência atual**; média de 6 exec., Seção 8.6) | **0,9934 ± 0,0021** | **0,9934 ± 0,0042** | **0,9933 ± 0,0060** | **0,9933 ± 0,0021** | **407,40 ± 17,09** | **2,29 ± 0,13** | **7,61 ± 0,43** | **0,002442 ± 0,000101** | **0,4348 ± 0,0246** | **130,88 ± 7,41** |

Os tempos (e, portanto, TES, IES e RTDE) só são comparáveis entre linhas medidas no mesmo hardware e em condições equivalentes: a linha do Colab usa outra máquina (Seção 7.5), e as linhas de 07/10 em diante usam outro conjunto de teste (Seção 1.1).

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
2. **As 6 execuções não são amostras independentes.** O recall tem desvio-padrão zero: 5 das 6 execuções erram exatamente os mesmos 2 e-mails (1 FP + 1 FN), e a execução 2 erra 1 FP a mais. As curvas de perda diferem entre execuções (não-determinismo do treino), mas o modelo final converge para as mesmas decisões, porque dados, split e semente do `Trainer` (42) são fixos. O ± mede só o não-determinismo de execução, não a incerteza do modelo — os p-valores acima devem ser lidos com essa ressalva (pseudo-replicação). Reforça a pendência de variar `random_state` (split) e `seed` (treino). *Decisão posterior (início da Fase 3):* manter o split e as sementes fixos no protocolo principal e tratar a variação de sementes no estudo de ablação (Seção 9.5).
3. **No protocolo antigo, a seleção no teste amplificava o ruído:** pequenas diferenças na trajetória mudavam a época escolhida, gerando variância maior (DP 0,0021 vs. 0,0014 na acurácia; resultado perfeito, 1,0000, na execução 2 de 25/09(b)). As curvas de 01/10 mostram o mecanismo — ex.: execução 4 com loss de avaliação ≈ 0,02 na época 3, ponto que o early stopping teria selecionado.
4. **Os 2 e-mails errados por todas as execuções são casos estruturalmente difíceis** — candidatos a análise de erros (rótulo ambíguo ou phishing com aparência legítima).
5. **Tempos.** Com épocas fixas, o tempo de treino fica estável (CV 2,7% vs. 12,7% em 25/09(b)) e o TES torna-se comparável entre execuções e entre modelos. A avaliação por época custa só ≈ 5,6% do treino, não explicando a diferença para o artigo. A diferença de tempo de **teste** entre os lotes (2,34 s vs. 3,92 s) não decorre do protocolo (a inferência é idêntica) — é variação de ambiente entre dias, já observada antes (25/09(a): 2,23 s); por isso, diferenças de IES/RTDE entre lotes não devem ser atribuídas a causas metodológicas. Dentro do próprio lote, as execuções 5 e 6 tiveram teste mais lento (2,72 s e 2,43 s), indicando máquina mais carregada.
6. **Contra o artigo, o padrão de tempos não é compatível com "hardware mais fraco" apenas:** o treino desta replicação é ≈ 2,2× mais lento, mas a inferência é ≈ 2,7× mais rápida (IES e RTDE maiores que os do artigo). Hipótese a investigar: configuração não documentada no artigo — comprimento máximo da sequência (aqui 128), batch de inferência, ou inclusão da tokenização no tempo de teste. O mesmo padrão (treino ≈ 2,3× mais lento, inferência ≈ 2,7× mais rápida) se repete no lote final de 07/10 (Seção 5.1).

---

## 4. Análise evolutiva

**17/09 → 19/09 (redução da escala do dataset e primeiro sinal do vazamento).** Reduzir o phishing de 1.501 para 501 amostras (para igualar a escala do artigo) derruba o recall (0,9915 → 0,9733) — esperado, menos dados de treino de phishing tende a piorar a detecção da classe minoritária. Mas a precisão já salta para **1,0000** nessa execução, mesmo sem os parâmetros anti-overfitting e com apenas 3 épocas. Em retrospecto, isso é o primeiro sinal do vazamento de dados que só seria formalmente diagnosticado na execução de 21/09 (Seção 5.5 do Embasamento Teórico): reduzir o pool de phishing a exatamente 501 amostras aumenta a proporção do dataset ocupada por conteúdo duplicado (a base do Nazario tem uma taxa de duplicação de ~4%, ver Seção 5.5), tornando mais provável que uma duplicata acabe ao mesmo tempo no treino e no teste. Com 1.501 amostras (17/09), o mesmo número absoluto de duplicatas fica diluído num pool maior, reduzindo a chance de uma dessas duplicatas cair especificamente nos dois lados do split.

**19/09 → 21/09 (mais épocas e proteção contra overfitting).** A precisão continua em 1,0000 — o vazamento não desaparece, porque os parâmetros de anti-overfitting (`early stopping`, `weight_decay`) atacam um problema diferente (o modelo memorizar padrões do próprio conjunto de treino) e não têm como corrigir duplicatas entre treino e teste. É essa persistência do artefato, já com um regime de treino mais cuidadoso, que motivou a investigação registrada na Seção 5.5.

**21/09 → 22/09 (primeira correção do vazamento, efetivamente executada).** Deduplicar o dataset já consolidado resolve o problema relatado — a precisão cai para um valor realista (0,9799, muito próximo do recall) — mas tem um efeito colateral não previsto: o alvo do balanceamento por undersampling passa a ser menor (481 em vez de 501, porque o Nazario perdeu 20 exemplares únicos na deduplicação), encolhendo o dataset de treino/teste e derrubando as quatro métricas preditivas (Acc./F1 de ~0,993 para ~0,979) e o TES (de 0,0048 para 0,0033).

**22/09 → 23/09 (correção do efeito colateral — desenhada, mas não confirmada em execução).** A ideia era deduplicar o pool bruto do Nazario *antes* de amostrar os 501 e-mails, resolvendo o vazamento sem encolher o dataset. O resultado observado (Acc. 0,9900, F1 0,9900, recall 0,9933) foi consistente com essa hipótese e, por isso, foi adotado como referência provisória — mas a Seção 1.2 registra que, em 24/09, se descobriu que a célula do notebook não chamava a função responsável por essa lógica, então o mecanismo real por trás desses números específicos não pôde ser confirmado.

**23/09 → 24/09 (reabertura da investigação: a causa raiz real).** Ao tentar estender a deduplicação prévia também ao lado dos legítimos, a precisão voltou a ficar suspeita. A investigação revelou dois bugs de implementação, não de lógica estatística: (1) a célula do Módulo A no notebook nunca invocava `amostrar_quantidade_artigo` — usava um trecho residual de 19/09 que amostrava o phishing manualmente sem deduplicação, tornando as edições do módulo `.py` feitas entre 21/09–23/09 código morto, nunca executado; (2) corrigido esse primeiro bug, uma segunda edição — para estender a deduplicação aos legítimos — trocou por engano quais listas alimentavam `legitimos_amostrados` e `phishing_amostrados`, invertendo os rótulos de classe do dataset inteiro. Ambos os bugs foram corrigidos e validados por log de execução no mesmo dia (Seção 5.6 do Embasamento Teórico traz o código antes/depois de cada correção).

**24/09 → 25/09(a) (primeiro pipeline realmente executado, ainda só com dedup do phishing).** Corrigido o Bug 1 (código morto do Módulo A), o lote 25/09(a) já roda a lógica pretendida — mas ainda só para o Nazario, já que a extensão da deduplicação ao SpamAssassin ainda não tinha sido implementada nesse ponto. Seis execuções repetidas mostraram Acc. 0,9939 ± 0,0033, Prec. 0,9989 ± 0,0027, Rec. 0,9889 ± 0,0054, F1 0,9939 ± 0,0033 — variância saudável (1 a 3 erros por execução), com os erros concentrados quase todos em falsos negativos do lado do phishing (só uma execução teve 1 falso positivo), padrão semelhante ao observado em 23/09.

**25/09(a) → 25/09(b) (extensão da deduplicação aos legítimos, novo bug e validação definitiva).** Ao estender a mesma lógica de deduplicação prévia também ao pool de legítimos (SpamAssassin), uma segunda edição introduziu o Bug 2 (troca de variáveis, rótulos de classe invertidos) — corrigido no mesmo dia. O lote final, 25/09(b), mostrou variância ainda mais saudável que o (a): Acc. 0,9967 ± 0,0021, Prec. 0,9978 ± 0,0034, Rec. 0,9956 ± 0,0034, F1 0,9967 ± 0,0021, com falsos positivos aparecendo pela primeira vez do lado dos legítimos (não só falsos negativos do lado do phishing, como em todas as execuções anteriores, incluindo 25/09(a)) — evidência adicional contra vazamento sistemático. Uma checagem programática explícita de sobreposição treino/teste confirmou, de forma direta (não apenas inferida da plausibilidade das métricas), a ausência total de vazamento: zero e-mails processados idênticos entre treino e teste, zero duplicatas internas em cada classe. Esse é o primeiro resultado da replicação cuja cadeia de execução foi verificada linha a código, e passou a ser a referência oficial (até 30/09 — Seção 1.3).

**Comparando 25/09(a) com 25/09(b): o efeito de deduplicar também o lado dos legítimos.** As quatro métricas preditivas melhoram de (a) para (b) — Acc. de 0,9939 para 0,9967, Prec. de 0,9989 para 0,9978 (leve queda, mas dentro do desvio-padrão de ambos os lotes), Rec. de 0,9889 para 0,9956, F1 de 0,9939 para 0,9967. O ganho mais consistente é no recall, que sobe de forma mais nítida entre os dois lotes — plausível, já que remover legítimos duplicados do pool de treino da classe majoritária tende a expor o modelo a mais conteúdo distinto de legítimos únicos, reduzindo a chance de o modelo "decorar" um padrão redundante e melhorando sua capacidade de distinguir os phishing verdadeiros. Já nas métricas de eficiência, o padrão é o oposto: TES, IES e RTDE caem de (a) para (b) (por exemplo, RTDE de 134,84 para 76,85), e o tempo médio de treino quase dobra (198,06 s para 352,24 s). Isso **não** pode ser atribuído ao tamanho do dataset — em ambos os lotes o undersampling (Módulo C) equaliza as duas classes ao tamanho da classe minoritária (501, o phishing já deduplicado em ambos os casos), então o dataset balanceado final tem exatamente 1.002 registros e ~301 no teste nos dois lotes, não importa se o pool bruto de legítimos tinha 2.501 ou 2.446 antes do corte. A diferença de tempo é, portanto, mais bem explicada por ruído de medição/ambiente entre as duas rodadas de execução (mesmo padrão de sensibilidade a variações de sistema já discutido na Seção 3.1 do Embasamento Teórico), não por uma diferença real de custo computacional entre as duas técnicas de deduplicação.

**25/09(b) → 01/10 (protocolo fiel ao artigo: remoção do viés de seleção no teste).** Com os mesmos dados de 25/09(b), o treino passou a seguir o protocolo do artigo: 10 épocas fixas, sem early stopping e sem recarregar o checkpoint de melhor F1 no teste. As métricas preditivas caíram de 0,9967 ± 0,0021 para 0,9928 ± 0,0014 — exatamente o tamanho da "vantagem" que 25/09(b) tinha sobre o artigo — e passaram a empatar com o artigo (0,9927). O tempo de treino ficou estável (CV 2,7% contra 12,7%), o que torna o TES comparável entre execuções e modelos. Lição: além do vazamento de **dados** (21–25/09), existe o vazamento por **seleção de modelo**, invisível à checagem de sobreposição treino/teste (Seção 3.4).

**01/10 → 02/10 (expansão para os cinco modelos).** O código passou a receber o modelo como parâmetro, com saídas organizadas por modelo. A reexecução do TinyBERT reproduziu exatamente o lote de 01/10 (validação da parametrização). O piloto (1 execução por modelo) repetiu a ordem de acurácia do artigo, mas mostrou tempos que crescem com o tamanho das arquiteturas — ao contrário da Tabela 3 do artigo — e duas instabilidades de fine-tuning: pico inicial de loss no MobileBERT e platô em ln 2 no ALBERT (Seção 6).

**02/10 → 03–04/10 (reprodutibilidade entre ambientes).** No Google Colab, a primeira execução divergiu (7 erros contra 2) apesar de contagens idênticas. A impressão digital do conteúdo mostrou que o dataset era outro: `glob` sem ordenação lia os arquivos em outra ordem no Linux. Corrigido com `sorted()`, o Colab reproduziu o PC local (0,9928 ± 0,0014 nos dois). Nos 5 modelos, 4 repetiram exatamente a acurácia local; o ALBERT passou de 9 para 3 erros (sem o platô), o que evidenciou a instabilidade do seu fine-tuning. Em hardware mais fraco, a vantagem de tempo do TinyBERT cresceu (Seção 7).

**04/10 (automação).** Criação do `rodar_lote.py`: um processo por treinamento, execução sequencial e registro automático de ambiente, versões e hashes dos dados em cada CSV — o que transforma as verificações manuais anteriores (contagens no log, hashes, renomeação de arquivos) em parte do próprio resultado (Seção 8.2).

**04/10 → 07/10 (validação do script e alinhamento do undersampling).** O lote de validação do script (TinyBERT, 501/501, 6 execuções) reproduziu a média de 01/10 (0,9928) e os hashes de referência. Em seguida, o undersampling foi alinhado à Figura 2 do artigo (502 + 501 = 1.003), o que mudou o conteúdo do teste (Seções 8.3 e 8.4).

**07/10 (lote final — referência atual).** Cinco modelos × 6 execuções, mesmos dados em todas. O TinyBERT (0,9934 ± 0,0021) e o DistilBERT (0,9928 ± 0,0014) empatam entre si e com o artigo; MobileBERT, ALBERT e MiniLM formam um grupo abaixo (4–6 erros em 301). O ALBERT não ficou preso no platô em nenhuma das 6 execuções e coincidiu com o artigo (0,9839 contra 0,9834); MiniLM e MobileBERT ficaram 2,5 e 1,4 e-mails abaixo do artigo. Os custos confirmam o piloto: o TinyBERT é 3,9–7,8× mais rápido no treino e 4,2–17,1× na inferência (Seção 8). Para o TinyBERT, a passagem 01/10 → 07/10 (0,9928 → 0,9934) mistura duas mudanças — outro conjunto de teste e outra rodada de treinos — e cabe dentro do erro-padrão (≈ ±0,5 p.p.); não deve ser lida como melhora.

---

## 5. Comparação final: resultado de referência (lote final de 07/10) vs. artigo original

Referência atual: lote final de 07/10/2026 — protocolo fiel ao artigo (10 épocas fixas, sem seleção no teste), undersampling 502/501 (1.003 amostras), split estratificado 70/30 (`random_state=42`), mesmos hiperparâmetros para todos os modelos (lr 5e-5, batch 8, `weight_decay=0.01`, `max_length=128`), 6 execuções por modelo, PC local (CPU Intel, 12 threads, sem GPU). Detalhes na Seção 8.

### 5.1 PHILDER (TinyBERT) vs. artigo — todas as métricas

| Métrica | Artigo (TinyBERT (2), Tab. 3 / Tab. 4; eficiência pela fórmula bruta) | Replicação (lote final, n = 6) | Diferença |
|---|---|---|---|
| Acurácia | 0,9927 ± 0,0006 | 0,9934 ± 0,0021 | +0,07 p.p. (≈ +0,2 e-mail) |
| Precisão | 0,9896 | 0,9934 ± 0,0042 | +0,38 p.p. |
| Recall | 0,9960 | 0,9933 ± 0,0060 | −0,27 p.p. |
| F1 | 0,9927 ± 0,0006 | 0,9933 ± 0,0021 | +0,06 p.p. |
| Erros em 301 (FP / FN) | — | 2,0 (1,0 / 1,0) | — |
| T. Treino bruto (s) | 180,52 | 407,40 ± 17,09 | 2,26× mais lento |
| T. Treino líquido (s) | — | 384,95 ± 16,81 | — |
| T. Teste (s) | 6,22 | 2,29 ± 0,13 | 2,72× mais rápido |
| Detecção (ms/amostra) | 12,9811 | 7,61 ± 0,43 | 1,71× mais rápido |
| TES (bruto) | 0,0055 | 0,002442 ± 0,000101 | 2,25× menor |
| TES (líquido) | — | 0,002585 ± 0,000111 | — |
| IES | 0,1596 | 0,4348 ± 0,0246 | 2,72× maior |
| RTDE | 76,47 | 130,88 ± 7,41 | 1,71× maior |

### 5.2 Os cinco modelos vs. artigo

| Modelo | Acc. artigo | Acc. replicação | Δ (e-mails em 301) | Rec. artigo | Rec. replicação | T. Treino artigo → replicação (s) | T. Teste artigo → replicação (s) | Treino × TinyBERT (artigo → replicação) | Teste × TinyBERT (artigo → replicação) |
|---|---|---|---|---|---|---|---|---|---|
| TinyBERT (PHILDER) | 0,9927 ± 0,0006 | **0,9934 ± 0,0021** | +0,2 | 0,9960 | 0,9933 | 180,52 → **407,4** | 6,22 → **2,29** | 1,00× → 1,00× | 1,00× → 1,00× |
| DistilBERT | 0,9934 ± 0,0005 | 0,9928 ± 0,0014 | −0,2 | 1,0000 | 0,9933 | 111,83 → 1.806,9 | 4,48 → 16,79 | 0,62× → 4,44× | 0,72× → 7,33× |
| MobileBERT | 0,9904 ± 0,0004 | 0,9856 ± 0,0034 | −1,4 | 0,9865 | 0,9800 | 219,45 → 1.581,4 | 9,76 → 11,11 | 1,22× → 3,88× | 1,57× → 4,85× |
| ALBERT | 0,9834 ± 0,0008 | 0,9839 ± 0,0044 | +0,2 | 0,9864 | 0,9911 | 236,81 → 3.170,0 | 9,09 → 39,11 | 1,31× → 7,78× | 1,46× → 17,07× |
| MiniLM | 0,9889 ± 0,0007 | 0,9806 ± 0,0039 | −2,5 | 0,9982 | 0,9822 | 103,52 → 1.574,3 | 6,21 → 9,63 | 0,57× → 3,86× | 1,00× → 4,20× |

### 5.3 Leitura

1. **Desempenho preditivo do PHILDER reproduzido.** Acurácia e F1 do TinyBERT ficam a +0,07 p.p. do artigo (≈ 0,2 e-mail em 301), bem dentro do erro-padrão binomial (≈ ±0,5 p.p.). Três dos cinco modelos (TinyBERT, DistilBERT e ALBERT) ficam a menos de meio e-mail das médias do artigo.
2. **A afirmação central do artigo se sustenta — com mais força do que no próprio artigo.** O TinyBERT empata com o modelo mais preciso (DistilBERT, diferença de 0,2 e-mail) e é o mais barato em todas as medidas de tempo: 3,9–7,8× mais rápido no treino e 4,2–17,1× na inferência. No artigo, pela Tabela 3, essa vantagem quase não aparece (MiniLM e DistilBERT treinariam mais rápido que o TinyBERT).
3. **Diferenças em relação ao artigo.** MiniLM (−2,5 e-mails) e MobileBERT (−1,4) ficaram abaixo do artigo, o que muda a ordem dos modelos intermediários (artigo: DistilBERT > TinyBERT > MobileBERT > MiniLM > ALBERT; replicação: TinyBERT ≈ DistilBERT > MobileBERT ≈ ALBERT > MiniLM). Entre esses três modelos as diferenças (1–1,5 e-mail) estão próximas do erro-padrão; a separação entre o grupo superior (≈ 2 erros) e o inferior (4–6 erros) é nítida. As causas prováveis — sobreajuste do MiniLM com 10 épocas fixas, erro sistemático do MobileBERT em 3 e-mails — estão na Seção 8.9.
4. **Eficiência: padrão consistente, mas diferente do artigo.** O treino é ≈ 2,3× mais lento e a inferência ≈ 2,7× mais rápida que no artigo, o mesmo padrão já visto em 01/10 (Seção 3.4, item 6). Isso não se explica só por "hardware mais fraco"; hipóteses: configuração não documentada no artigo (comprimento de sequência, batch de inferência, inclusão da tokenização no tempo de teste) e/ou uso de GPU pelos autores. As métricas de eficiência só são comparáveis no mesmo hardware (Seção 7.5).
5. **Ressalva estatística.** As 6 execuções variam apenas o não-determinismo do treino, com dados e teste fixos: os desvios-padrão medem a estabilidade do fine-tuning, não a incerteza sobre dados novos. Diferenças de 1–2 e-mails entre modelos não devem ser declaradas significativas.

### 5.4 Registro histórico — comparação anterior (25/09(b) vs. artigo)

Comparação feita em 25/09, mantida para registro. Está superada: usava valores das Figuras 3–5 do artigo (inconsistentes com a sua Tabela 3) e um lote com seleção de modelo no teste.

> **Atualização 01/10/2026.** Esta comparação foi escrita antes de duas correções: (1) os tempos e as métricas de eficiência do artigo usados abaixo vinham das Figuras 3–5 e de uma linha errada da Tabela 3 (ver Seção 2 — valores corrigidos: TES 0,0055, IES 0,1596, RTDE 76,47); (2) o lote 25/09(b) usa o conjunto de teste para seleção de modelo. A comparação com o protocolo fiel ao artigo está na Seção 3.4. **Atualização 08/10/2026:** a redefinição do resultado de referência, pendente em 01/10, foi concluída — referência atual é o lote final de 07/10 (Seções 5.1–5.3).

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

## 6. Expansão dos modelos avaliados (Fase 3 — piloto de 02/10/2026)

### 6.1 Contexto da inclusão dos modelos

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

### 6.2 Ajuste do código

O pipeline (Módulos A–D) e o protocolo de treino **não mudaram** — mesmos dados (amostragem, deduplicação, undersampling e split com `random_state=42`), protocolo "artigo" (10 épocas fixas, sem early stopping e sem seleção de checkpoint no teste) e mesmos hiperparâmetros para todos os modelos (lr 5e-5, batch 8, `weight_decay=0.01`, `max_length=128`). A mudança foi apenas na forma de **escolher o modelo e organizar as saídas**:

- **Notebook — nova célula de configuração:** dicionário `modelos_artigo` (apelido → checkpoint), variável `modelo_escolhido` como único ponto de troca entre experimentos, pastas derivadas (`pasta_modelo`, `pasta_checkpoints`, `pasta_resultados`) e a função `proxima_rodada`, que numera automaticamente as execuções do dia (maior rodada existente + 1), eliminando a renomeação manual de arquivos.
- **Notebook — células C e D:** o checkpoint passa a ser enviado a `carregar_tokenizer_e_modelo(checkpoint_modelo)`; o modelo vai para `modelo/<modelo>/` (checkpoints em `modelo/<modelo>/checkpoints/`) e os resultados para `resultados/<modelo>/resultado_final-<rodada>_<ddmmaaaa>.csv` e `curva_perda-<rodada>_<ddmmaaaa>.png`. O CSV ganhou as colunas de rastreabilidade `modelo`, `checkpoint` e `rodada`. A variável `modelo_tb` foi renomeada para `modelo_pre_treinado`.
- **Módulo D (`performance_assessment.gerar_curva_perda`):** novo parâmetro opcional `titulo`, para identificar o modelo no gráfico (padrão mantém o título anterior).
- **Procedimento de execução:** kernel reiniciado entre um modelo e outro (memória limpa e tempos mais comparáveis), com conferência no log dos números de controle do dataset (2.947 registros — 2.446 legítimos, 501 phishing; treino {1: 351, 0: 350}; teste {0: 151, 1: 150}). A partir de 04/10, esse procedimento foi automatizado pelo `rodar_lote.py` (Seção 8.2).

**Validação do código novo.** O TinyBERT foi reexecutado com o código parametrizado (02/10, rodada 1) e reproduziu exatamente o resultado do lote de 01/10: Acurácia 0,993355 (1 FP + 1 FN), idêntica a 5 das 6 execuções anteriores, com curva de perda equivalente. Isso confirma que a parametrização não alterou o pipeline e valida, por extensão, as execuções dos demais modelos.

> **Registro de incidente operacional (01–02/10) — corrigido em 02/10.** Durante esta etapa, o notebook e o módulo `performance_assessment.py` voltaram, sem aviso, a versões anteriores, desfazendo alterações já aplicadas (o efeito visível foi um `TypeError` por parâmetro inexistente). A primeira hipótese registrada — abas antigas do Jupyter Lab salvando por cima — estava **errada**. A causa real foi o processo de gravação dos arquivos a partir da sessão do assistente (Claude): ao reaproveitar o mesmo caminho temporário de um envio anterior e enviar logo em seguida, o computador recebia a versão **anterior** daquele caminho (confirmado em 02/10: o arquivo gravado era idêntico, byte a byte, ao envio anterior, com horário de modificação igual ao da nova gravação). Correção adotada: caminho temporário único a cada envio e conferência de toda gravação por leitura do arquivo de volta (comparação de conteúdo). Notebook e módulo foram restaurados e conferidos. Lição para a reprodutibilidade: uma gravação "concluída sem erro" não prova que o conteúdo pretendido chegou ao destino — só a verificação direta do arquivo resultante prova (mesmo princípio da lição central do vazamento de dados, Seção 1.2), e cada estado validado deve ser versionado (commit no Git).

### 6.3 Tabela consolidada de resultados (piloto — 1 execução por modelo)

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

### 6.4 Achados do piloto

1. **A ordenação de acurácia é idêntica à do artigo.** DistilBERT > TinyBERT > MobileBERT > MiniLM > ALBERT, tanto na Tabela 4 do artigo quanto nesta replicação — com dataset substituto, sem código dos autores e com hiperparâmetros não especificados no artigo. TinyBERT (0,9934 vs. 0,9927) e MobileBERT (0,9900 vs. 0,9904) ficam a menos de 1 e-mail do artigo; MiniLM a ~1 e-mail. O DistilBERT repete inclusive o recall 1,0000 do artigo. Ressalva: com n = 301, o erro-padrão da acurácia é ≈ ±0,5 p.p. (≈ 1,5 e-mail), de modo que diferenças de 1–2 e-mails entre modelos não são distinguíveis com uma execução. *Atualização 07/10:* no lote final (6 execuções, 502/501), a ordem dos modelos intermediários mudou — MiniLM e MobileBERT abaixo do artigo, ALBERT igual ao artigo (Seção 8.9).
2. **A tese do PHILDER se confirma com mais força nas medições próprias do que no próprio artigo.** O TinyBERT é 3,6–7,7× mais rápido no treino e 4,2–16,7× mais rápido na inferência que os demais, e a ordem de custo acompanha o tamanho das arquiteturas. A diferença de acurácia em relação ao melhor modelo (DistilBERT) é de 1 e-mail, ao custo de 4,3× no treino e 7,1× na inferência — exatamente o trade-off que TES, IES e RTDE foram desenhados para expor.
3. **A Tabela 3 do artigo não é compatível com o custo das arquiteturas.** No artigo, os tempos variam numa faixa estreita (≈ 0,6–1,3× no treino, 0,7–1,6× no teste, em relação ao TinyBERT) e a ordem se inverte em relação ao tamanho dos modelos (MiniLM e DistilBERT mais rápidos que o TinyBERT). **Hipótese (não confirmada):** medição em GPU, onde modelos pequenos não saturam o hardware e os tempos passam a ser dominados por custos fixos (transferência de dados, carregamento, sincronização), comprimindo e embaralhando as diferenças. Em CPU — cenário mais próximo de um dispositivo de borda, o alvo do PHILDER — o custo computacional domina e a vantagem do TinyBERT aparece integralmente. Se confirmada, a vantagem do PHILDER seria **maior** justamente no ambiente para o qual foi proposto.
4. **MobileBERT — instabilidade inicial.** A loss de treino começa em ≈ 1,7 × 10⁶ e cai a ≈ 0 em menos de 20 passos; o modelo se recupera (acurácia alinhada ao artigo). Comportamento conhecido da arquitetura (camadas "NoNorm", sem LayerNorm, gerando logits muito grandes com a camada de classificação recém-inicializada), contido pelo gradient clipping padrão do `Trainer`. Consequência prática: a curva de perda em escala linear fica ilegível para este modelo.
5. **ALBERT — platô inicial e menor convergência.** A loss permaneceu em ≈ 0,69 (= ln 2, equivalente a um classificador que chuta 50/50) nas primeiras ~4 épocas, só passando a aprender a partir da 5ª; ao final das 10 épocas a loss de avaliação (≈ 0,17) ainda era bem maior que a dos demais (≈ 0,03–0,09). Isso explica a acurácia mais baixa e o maior afastamento em relação ao artigo (0,9701 vs. 0,9834). O ALBERT é conhecido por exigir learning rate menor e warmup no fine-tuning; **os hiperparâmetros foram deliberadamente mantidos iguais para todos os modelos** — ajustá-los por modelo a partir do resultado no teste reintroduziria o viés de seleção eliminado em 01/10 (Seção 3.4). Registrado como limitação. *Atualização 07/10:* no lote final, nenhuma das 6 execuções do ALBERT ficou no platô (Seção 8.9, item 4).
6. **Custo de avaliação por época** varia de 5,6% (TinyBERT) a 12,4% (ALBERT) do tempo de treino bruto; o TES líquido não altera a ordenação dos modelos.

### 6.5 Viabilidade e próximos passos

- **Tempo de uma rodada completa (5 modelos):** ≈ 2h10 de treino (≈ 2,5 h com carregamento e teste). Como o TinyBERT já tem 6 execuções, completar os outros quatro modelos custa ≈ 2h20 por rodada adicional: ≈ 4h40 para chegar a 3 execuções por modelo (como o artigo) e ≈ 11h40 para 6.
- ~~**Pendente de decisão:** número de execuções por modelo na comparação final (3 ou 6).~~ Decidido: 6 execuções por modelo (lote final de 07/10, Seção 8). Duração real do lote completo (5 modelos × 6 execuções): ≈ 14,5 h no PC local (09h06–23h39 de 07/10).
- Avaliar escala logarítmica (ou recorte dos passos iniciais) em `gerar_curva_perda`, para tornar legíveis curvas como a do MobileBERT.
- Análise de erros: inspecionar os e-mails classificados incorretamente (em especial o phishing perdido por TinyBERT, MiniLM e MobileBERT e acertado pelo DistilBERT).
- Os tempos dependem do estado da máquina (ex.: TinyBERT 382 s em 02/10 vs. 406 ± 11 s em 01/10, mesmo código): as rodadas finais devem ser executadas em condições controladas e equivalentes para todos os modelos. *Atendido em parte* pelo `rodar_lote.py` (execução sequencial, um processo por treinamento); a variação de carga da máquina ainda aparece no lote final (Seção 8.9, item 7).

---

## 7. Reprodutibilidade entre ambientes (Google Colab — 03/10/2026)

### 7.1 Contexto

Para permitir o trabalho remoto (sem o computador local), o mesmo notebook e o mesmo pacote `philder/` foram executados no **Google Colab**, a partir de uma cópia do projeto no Google Drive. O objetivo foi verificar se o pipeline produz os mesmos resultados num ambiente diferente — outro sistema operacional, outro processador e outras versões de bibliotecas — com o mesmo código, os mesmos dados de origem e as mesmas sementes.

| | PC local | Google Colab |
|---|---|---|
| Sistema operacional | Windows | Linux |
| Processador | CPU do notebook local | 2 núcleos virtuais AMD EPYC 7B12 (compartilhado), sem GPU |
| Python | 3.12.13 | 3.13.15 |
| torch | 2.14.0+cpu | 2.11.0+cpu |
| transformers | 5.16.1 | 5.17.0 |

### 7.2 Falha de reprodutibilidade identificada: ordem de leitura dos arquivos

**Sintoma.** A primeira execução do TinyBERT no Colab obteve Acurácia 0,9767 (**7 erros**: 5 FP + 2 FN), contra 0,9934 (2 erros: 1 FP + 1 FN) no PC local — um salto grande demais para ser explicado por ruído numérico entre máquinas. As contagens de controle no log eram **idênticas** nos dois ambientes (2.947 registros — 2.446 legítimos, 501 phishing; treino {1: 351, 0: 350}; teste {0: 151, 1: 150}) e não indicavam nenhuma diferença.

**Diagnóstico.** Foi calculada uma impressão digital (hash MD5) do **conteúdo** dos conjuntos — os textos de `body_processado`, ordenados e concatenados — nos dois ambientes, após o Módulo C:

| Conjunto | PC local | Colab (antes da correção) |
|---|---|---|
| Balanceado | `c818699ea3118932f9a64f9dcc16169e` | `b10ef31410509e5ab6ec325452f6f811` |
| Treino | `c7f1187377e71ac94aa2f100ee2d3e08` | `f9c9cc874ed13f19ac3ad186fa4a1ced` |
| Teste | `243d7b22075cf4f6d88ecc63c04005ed` | `b6b8d650b77e7c78ddac5da4c945b6f9` |

Os três conjuntos diferiam. A causa estava em `importar_corpus_spamassassin` (Módulo A), que lê os e-mails com `glob.glob(...)` **sem ordenação**: no Windows (NTFS) a lista vem em ordem alfabética (`00001…`, `00002…`, `00003…`); no Linux do Colab, em ordem arbitrária (`02159…`, `02058…`, `00737…`). Essa ordem se propaga pelo pipeline: (1) a deduplicação (`keep="first"`) escolhe outro representante entre duplicatas; (2) o DataFrame consolidado fica em outra ordem e o undersampling, com a mesma semente, sorteia **outros 501 legítimos**; (3) o split 70/30, com a mesma semente, gera **outro conjunto de teste**. O resultado de 0,9767 foi, portanto, medido sobre **outro dataset** e foi descartado.

**Correção.** `arquivos_ham = sorted(glob.glob(os.path.join(pasta_extraida, "*")))`. Como o Windows já devolvia os arquivos em ordem alfabética, a ordenação reproduz exatamente a ordem sempre usada no PC local — **nenhum resultado local anterior é afetado** — e torna a leitura determinística em qualquer sistema. Após a correção, os hashes do Colab passaram a coincidir com os do PC local. *Situação em 03/10:* correção aplicada na cópia do Colab e, no mesmo dia, incorporada ao código do PC local (`philder/database_consolidation.py`, com comentário datado e a linha anterior preservada como comentário). Pendentes, na data: commit/push para o repositório GitHub e verificação, no PC local, de que os três hashes permanecem iguais. *Atualização:* a verificação no PC local foi feita em 03/10 (hashes inalterados) e reconfirmada em 07/10 pelo lote de validação do `rodar_lote.py`, que registra os hashes automaticamente (Seção 8.3).

**Lição para a reprodutibilidade.** Sementes fixas não garantem reprodutibilidade se a **ordem de entrada dos dados** depender do sistema operacional. Contagens idênticas não provam conteúdo idêntico — só a verificação direta do conteúdo (impressão digital) revelou a diferença. É o mesmo princípio das lições anteriores do projeto (vazamento de dados, Seção 1.2; verificação de gravação, Seção 6.2).

### 7.3 Resultados no Colab após a correção (TinyBERT, 6 execuções)

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

### 7.4 Comparação entre ambientes

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

### 7.5 Leitura

1. **O desempenho preditivo se reproduz entre ambientes.** Com outro sistema operacional, outro processador e outras versões de Python, torch e transformers, Acurácia e F1 têm média e desvio-padrão iguais até a 4ª casa decimal, e o mesmo padrão de erros se repete (5 de 6 execuções com exatamente 1 FP + 1 FN). A única diferença é a execução fora do padrão (um FN a mais no Colab, um FP a mais no PC local) — variação do tamanho de um e-mail na fronteira de decisão. Isso reforça a confiabilidade do resultado da replicação.
2. **A diferença inicial vinha inteiramente dos dados, não do ambiente.** Antes da correção da ordem de leitura, o mesmo código no Colab produziu 7 erros; depois dela, 2.
3. **Os tempos escalam com o hardware, e de forma desigual entre as métricas.** No Colab (2 núcleos virtuais compartilhados), o treino foi ≈ 1,6× mais lento e o teste ≈ 2,6× mais lento que no PC local; a inferência em lote perde mais com poucos núcleos. A avaliação por época também pesa mais (9,8% vs. 5,6% do treino). Consequência: **as métricas de eficiência não são transportáveis entre ambientes por proporção simples** — TES, IES e RTDE só são comparáveis quando medidos no mesmo hardware, e o hardware precisa ser reportado junto com elas.
4. **Coincidência a ser lida com cautela.** O tempo de teste e o IES do Colab (6,20 s; 0,161) ficaram quase iguais aos do artigo (6,22 s; 0,160), enquanto o treino do artigo (180,52 s) é ≈ 3,5× mais rápido que o do Colab. Esses valores não se encaixam num único hardware conhecido, o que reforça que a Tabela 3 do artigo não admite explicação simples (Seção 6.4); nada se pode concluir sobre o hardware dos autores a partir disso.

### 7.6 Pendências

- ~~Incorporar `sorted(glob.glob(...))` ao código do PC local~~ (feito em 03/10). ~~Verificar no PC local que os três hashes permanecem iguais (`c818699e…`, `c7f11873…`, `243d7b22…`)~~ (feito em 03/10; reconfirmado em 07/10, Seção 8.3). Falta: commit/push para o GitHub.
- ~~Registrar, em toda execução fora do PC local, o hardware e as versões de Python/torch/transformers junto com os resultados~~ — automatizado em 04/10: o `rodar_lote.py` grava plataforma, processador, núcleos, GPU e versões em cada CSV (Seção 8.2). Resultados do Colab continuam separados dos do PC local.
- Avaliar fixar as versões das bibliotecas (`requirements.txt`) no pacote de reprodutibilidade.

### 7.7 Expansão no Colab: os 5 modelos (1 execução por modelo, 03–04/10/2026)

Mesmo ambiente da Seção 7.1 (Colab, CPU com 2 núcleos virtuais, sem GPU), mesmo código (com a ordenação da Seção 7.2), dados e protocolo. Uma execução por modelo (TinyBERT: execução 6 de 03/10; MiniLM: execução 2 de 03/10; demais: 04/10).

**Métricas preditivas — Colab vs. PC local (piloto de 02/10, Seção 6.3)**

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

**Leitura.** Quatro dos cinco modelos repetem exatamente a acurácia e o número de erros do PC local (DistilBERT e TinyBERT com os mesmos tipos de erro; MiniLM e MobileBERT com troca de um FP por um FN — ruído de fronteira). É mais uma confirmação de que os dados são idênticos nos dois ambientes (Seção 7.2). A ordem de acurácia no Colab — DistilBERT > TinyBERT > MobileBERT = ALBERT > MiniLM — difere da do artigo e do PC local apenas na posição do ALBERT (Seção 7.8). Viabilidade: uma rodada completa dos 5 modelos levou ≈ 6,5 h no Colab (ALBERT sozinho > 3 h), o que torna arriscado rodar lotes completos na versão gratuita (sessão máxima e desconexão por inatividade).

### 7.8 Instabilidade de fine-tuning do ALBERT

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

> **Atualização 07/10/2026.** No lote final (502/501, PC local, 6 execuções), **nenhuma** execução do ALBERT ficou no platô: em todas, a loss de treino cai já na 1ª época, e a loss de avaliação final ficou entre ≈ 0,08 e 0,14. Os picos fortes da loss de treino persistem (até ≈ 0,6–0,7). Acurácia 0,9839 ± 0,0044 (3–6 erros), igual à média do artigo. Contagem de platôs até aqui: 1 de 1 no piloto local (02/10, dados 501/501), 0 de 1 no Colab (04/10, 501/501), 0 de 6 no lote final (502/501). O platô é, portanto, um desfecho possível mas não sistemático; como o dado também mudou entre o piloto e o lote final, não é possível separar o efeito do dado do efeito do acaso.

### 7.9 Efeito do hardware na vantagem do TinyBERT

Lentidão do Colab (2 núcleos virtuais compartilhados) em relação ao PC local, por modelo:

| Modelo | Dimensão interna | Treino (Colab ÷ PC) | Teste (Colab ÷ PC) |
|---|---|---|---|
| TinyBERT | 312 | 1,65× | 2,90× |
| MiniLM | 384 | 1,57× | 2,57× |
| MobileBERT | 128 (gargalo) / 512 | 2,69× | 3,87× |
| DistilBERT | 768 | 3,51× | 4,20× |
| ALBERT | 768 | 3,89× | 4,03× |

Os modelos com dimensão interna 768 foram os mais penalizados pela máquina mais fraca. **Hipótese (não verificada):** as matrizes dos modelos menores cabem melhor na memória cache do processador, enquanto as dos maiores dependem mais da memória principal — gargalo típico de uma máquina virtual compartilhada.

Consequência: a vantagem de tempo do TinyBERT sobre os demais modelos passa de 3,6–7,7× no treino e 4,2–16,7× na inferência (PC local, Seção 6.4) para **3,6–18,1× no treino e 3,7–23,3× na inferência** (Colab). **Quanto mais limitado o hardware, maior a vantagem do PHILDER** — evidência, em dois ambientes de CPU, a favor da tese do artigo para dispositivos com recursos limitados, e coerente com a hipótese levantada para a Tabela 3 (Seção 6.4): em hardware potente (GPU), as diferenças entre os modelos tenderiam a encolher.

### 7.10 Pendências adicionais

- ~~Nas rodadas múltiplas finais, registrar para o ALBERT se houve platô inicial, a partir da curva de perda de cada execução~~ — feito para o ALBERT em 07/10 (0 de 6; Seção 7.8 e Seção 8.9). Falta conferir as curvas dos demais modelos do lote final.
- Avaliar, como experimento complementar e explicitamente separado do protocolo principal, o efeito de warmup/learning rate menor sobre a estabilidade do ALBERT — sem usar o resultado no teste para escolher a configuração do protocolo principal.
- Experimento GPU vs. CPU (Colab/Kaggle com GPU) para testar a hipótese de compressão das diferenças de tempo em hardware potente.

---

## 8. Automação, alinhamento do undersampling e lote final (04–07/10/2026)

### 8.1 Contexto e sequência de ações

Depois do piloto (Seção 6) e da reprodução no Colab (Seção 7), faltavam três coisas para a comparação final entre os modelos: (i) executar muitas rodadas sem intervenção manual e com condições equivalentes; (ii) alinhar o tamanho do dataset balanceado ao do artigo; e (iii) obter 6 execuções de cada modelo nas mesmas condições. A sequência foi:

| Data | Ação | Seção |
|---|---|---|
| 04/10 | Criação do `rodar_lote.py` (execução em lote, um processo por treinamento) | 8.2 |
| 07/10, manhã | Validação do script: TinyBERT, 501/501, 6 execuções | 8.3 |
| 07/10 | Undersampling alinhado ao artigo (502/501) e validação de 1 execução no notebook | 8.4 |
| 07/10, 09h06–23h39 | Lote final: 5 modelos × 6 execuções, 502/501 | 8.5–8.10 |

### 8.2 Automação da execução: `rodar_lote.py` (04/10; ajuste em 07/10)

Script independente na raiz do projeto, que reaproveita os módulos do pacote `philder/` (A–D) na mesma sequência de chamadas do notebook, sem alterá-los. O notebook continua utilizável e não foi modificado para isso.

- **Um processo por treinamento.** O orquestrador chama o próprio script em modo `--trabalhador`, num processo Python novo para cada par (modelo, rodada). Ao terminar, o processo libera toda a memória — o equivalente automático a "reiniciar o kernel" (procedimento manual da Seção 6.2).
- **Execução sequencial, nunca em paralelo,** para que cada treinamento use a máquina inteira e os tempos (TES/IES/RTDE) fiquem comparáveis.
- **Opções:** `--modelos` (nomes ou `todos`), `--rodadas` (por modelo), `--simular` (mostra o plano sem treinar) e, desde 07/10, `--undersampling` (`artigo` = 502/501, padrão; `minoritaria` = 501/501; ou `L,P`).
- **Saídas** no mesmo padrão do notebook: `resultados/<modelo>/resultado_final-<rodada>_<data>.csv`, `curva_perda-<rodada>_<data>.png`, `log_execucao-<rodada>_<data>.txt` (saída completa do processo) e um resumo `resultados/lote_<data>_<hora>.csv` (ordem, modelo, rodada, status, duração do processo, log), regravado a cada passo.
- **Rastreabilidade em cada CSV:** além das colunas do notebook, `origem`, `plataforma`, `processador`, `nucleos`, `gpu`, versões de `python`/`torch`/`transformers`, as impressões digitais MD5 dos conjuntos balanceado, treino e teste (`hash_balanceado`, `hash_treino`, `hash_teste` — a verificação que revelou o problema do `glob`, Seção 7.2) e, desde 07/10, `undersampling`, `n_balanceado_legitimos`, `n_balanceado_phishing`, `n_treino`, `n_teste`.
- **Numeração das rodadas:** continua a do dia (maior rodada existente + 1) e considera também os logs de tentativas que falharam, para nunca sobrescrever o log de uma tentativa anterior. Esta segunda regra foi uma correção feita durante o desenvolvimento: na primeira versão, uma rodada que falhava reaproveitava o número e o log.

### 8.3 Validação do script: lote TinyBERT 501/501 (07/10, manhã)

Antes de mudar qualquer outra coisa, o script foi validado com a configuração já conhecida: TinyBERT, 501/501, protocolo "artigo", PC local. A rodada 1 do dia foi feita pelo notebook (Acc. 0,9934; 1 FP + 1 FN; treino 448,02 s; teste 2,39 s); as rodadas 2–7 pelo script (resumo `resultados/lote_07102026_072428.csv`, 6 execuções sem falha). Os arquivos foram depois movidos para `resultados/tinybert/teste e validacao/` (uma cópia da rodada 7 permaneceu em `resultados/tinybert/`).

| Execução | Acc. | Prec. | Rec. | F1 | FP | FN | T. Treino bruto (s) | T. Aval. por época (s) | T. Treino líquido (s) | T. Teste (s) | Detecção (ms) | TES (bruto) | TES (líquido) | IES | RTDE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 440,83 | 23,20 | 417,63 | 2,34 | 7,79 | 0,002253 | 0,002379 | 0,4238 | 127,56 |
| 3 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 438,92 | 23,18 | 415,74 | 2,32 | 7,72 | 0,002263 | 0,002389 | 0,4273 | 128,63 |
| 4 | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 2 | 1 | 442,59 | 23,33 | 419,26 | 2,47 | 8,19 | 0,002237 | 0,002361 | 0,4015 | 120,84 |
| 5 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 442,27 | 23,40 | 418,86 | 2,31 | 7,67 | 0,002246 | 0,002372 | 0,4300 | 129,44 |
| 6 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 0 | 1 | 440,05 | 25,21 | 414,84 | 2,31 | 7,68 | 0,002265 | 0,002403 | 0,4312 | 129,78 |
| 7 | 0,9900 | 0,9933 | 0,9867 | 0,9900 | 1 | 2 | 461,53 | 24,56 | 436,98 | 2,34 | 7,77 | 0,002145 | 0,002266 | 0,4235 | 127,47 |
| **Média ± DP** | **0,9928 ± 0,0025** | **0,9933 ± 0,0042** | **0,9922 ± 0,0027** | **0,9928 ± 0,0025** | — | — | **444,37 ± 8,52** | **23,81 ± 0,86** | **420,55 ± 8,23** | **2,35 ± 0,06** | **7,80 ± 0,20** | **0,002235 ± 0,000045** | **0,002361 ± 0,000049** | **0,4229 ± 0,0109** | **127,29 ± 3,29** |

**Leitura.** (1) Os três hashes das 6 execuções são os de referência 501/501 (`c818699e…`, `c7f11873…`, `243d7b22…`): o script usa exatamente os mesmos dados do notebook, e a correção do `glob` (Seção 7.2) se mantém no PC local. (2) A acurácia média é a mesma do lote de 01/10 (0,9928). (3) O treino ficou ≈ 38 s mais lento que em 01/10 (444 s contra 406 s), com teste igual (2,35 s contra 2,34 s) — variação de estado da máquina entre dias, não do script. Resultado: script validado.

### 8.4 Alinhamento do undersampling ao artigo (502/501) e validação no notebook (07/10)

**Motivação.** A Figura 2 do artigo mostra o dataset balanceado com **502 legítimos + 501 phishing = 1.003** amostras. Até aqui, o undersampling igualava as classes no tamanho da minoritária (501/501 = 1.002). A diferença é de uma amostra, mas muda o sorteio do split e, portanto, o conteúdo do teste.

**Alterações de código (aprovadas uma a uma):**

- `philder/database_balance.py` — `aplicar_undersampling(df, coluna_rotulo="class_label", random_state=42, n_por_classe=None)`: com `None`, mantém o comportamento anterior (todas as classes no tamanho da minoritária); com um dicionário, usa o alvo de cada classe, e gera erro (`ValueError`) se faltar alguma classe ou se o alvo for maior que o disponível. A mensagem de log passou a informar o alvo usado.
- Notebook — célula de configuração: `undersampling_alvo = {0: 502, 1: 501}`; célula C: `aplicar_undersampling(df, n_por_classe=undersampling_alvo)`; célula D: nova coluna `undersampling` no CSV de resultados.
- `rodar_lote.py` — opção `--undersampling` (Seção 8.2), com padrão `artigo` (502/501).

**Validação (1 execução do TinyBERT pelo notebook):**

| Acc. | Prec. | Rec. | F1 | FP | FN | T. Treino bruto (s) | T. Aval. por época (s) | T. Treino líquido (s) | T. Teste (s) | Detecção (ms) | TES (bruto) | TES (líquido) | IES | RTDE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0,9900 | 0,9804 | 1,0000 | 0,9901 | 3 | 0 | 424,11 | 22,57 | 401,53 | 2,35 | 7,82 | 0,002334 | 0,002466 | 0,4205 | 126,56 |

A coluna `undersampling` registrou `{0: 502, 1: 501}`. Pelas métricas, o teste tem 150 phishing (recall 1,0 com precisão 150/153) e 151 legítimos — o esperado do split estratificado de 1.003 registros. A curva de perda não mostrou anomalias (perda de avaliação final ≈ 0,05, com pico ≈ 0,105 nas épocas 4–5). Os 3 erros (todos FP) ficam dentro do erro-padrão e não são comparáveis e-mail a e-mail com os resultados 501/501, porque o teste é outro. *Nota de arquivo:* o CSV desta execução (rodada 2 de 07/10, gerado pelo notebook) não está mais em `resultados/tinybert/` — a numeração do lote final recomeçou na rodada 1 —; resta uma cópia em `resultados/tinybert/.ipynb_checkpoints/resultado_final-2_07102026-checkpoint.csv`.

### 8.5 Lote final — configuração e dados de referência

- **Comando:** `python rodar_lote.py --modelos todos --rodadas 6` (undersampling padrão `artigo`, 502/501). Resumo: `resultados/lote_07102026_090610.csv`.
- **Execução:** 30 treinamentos, todos com status "ok", um processo por treinamento, sem paralelismo, das 09h06 às 23h39 de 07/10 (≈ 14,5 h). Ordem: TinyBERT, MiniLM, DistilBERT, MobileBERT, ALBERT (6 rodadas de cada, em sequência).
- **Ambiente (igual nas 30 execuções):** PC local, Windows, "Intel64 Family 6 Model 158" (12 threads), sem GPU; Python 3.12.13, torch 2.14.0+cpu, transformers 5.16.1.
- **Protocolo e hiperparâmetros:** protocolo "artigo" (10 épocas fixas, sem early stopping e sem seleção de checkpoint; avaliação por época só para a curva de perda); lr 5e-5, batch 8, `weight_decay=0.01`, `max_length=128`, iguais para todos os modelos; split estratificado 70/30, `random_state=42`.
- **Dados de referência (impressão digital MD5, idêntica nas 30 execuções):**

| Conjunto | Tamanho | Hash |
|---|---|---|
| Balanceado | 1.003 (502 ham / 501 phishing) | `b5eed2f9bf4f8dd5bd32fe182f56a8c9` |
| Treino | 702 (351 / 351) | `d7ee416586b457c1696481c80eeef4cc` |
| Teste | 301 (151 ham / 150 phishing) | `c2aecb1bdebe9d1fc4b98740e7e08da6` |

Os hashes 501/501 anteriores (`c818699e…`, `c7f11873…`, `243d7b22…`) passam a identificar o conjunto antigo. O arquivo `resultados/tinybert/resultado_final-7_07102026.csv` é a cópia da rodada 7 do lote de validação 501/501 (Seção 8.3) e foi **excluído** desta consolidação (hashes diferentes). Como o conteúdo do teste mudou, os resultados desta seção não são comparáveis e-mail a e-mail com os das Seções 3–7.

### 8.6 Tabela consolidada (média ± desvio-padrão, n = 6 por modelo)

| Modelo | Acurácia | Precisão | Recall | F1 | Erros (FP / FN) | T. treino (s) | T. teste (s) | Detecção (ms) | TES | IES | RTDE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **TinyBERT (PHILDER)** | **0,9934 ± 0,0021** | 0,9934 ± 0,0042 | 0,9933 ± 0,0060 | 0,9933 ± 0,0021 | 2,0 (1,0 / 1,0) | **407,4 ± 17,1** | **2,29 ± 0,13** | **7,61 ± 0,43** | **0,00244 ± 0,00010** | **0,435 ± 0,025** | **130,9 ± 7,4** |
| DistilBERT | 0,9928 ± 0,0014 | 0,9923 ± 0,0050 | 0,9933 ± 0,0042 | 0,9928 ± 0,0013 | 2,2 (1,2 / 1,0) | 1.806,9 ± 4,9 | 16,79 ± 0,06 | 55,77 ± 0,19 | 0,00055 ± 0,00000 | 0,059 ± 0,000 | 17,8 ± 0,1 |
| MobileBERT | 0,9856 ± 0,0034 | 0,9911 ± 0,0069 | 0,9800 ± 0,0000 | 0,9855 ± 0,0034 | 4,3 (1,3 / 3,0) | 1.581,4 ± 65,6 | 11,11 ± 0,04 | 36,90 ± 0,14 | 0,00062 ± 0,00003 | 0,089 ± 0,000 | 26,7 ± 0,1 |
| ALBERT | 0,9839 ± 0,0044 | 0,9771 ± 0,0079 | 0,9911 ± 0,0054 | 0,9840 ± 0,0044 | 4,8 (3,5 / 1,3) | 3.170,0 ± 40,3 | 39,11 ± 0,52 | 129,95 ± 1,72 | 0,00031 ± 0,00000 | 0,025 ± 0,000 | 7,6 ± 0,1 |
| MiniLM | 0,9806 ± 0,0039 | 0,9790 ± 0,0064 | 0,9822 ± 0,0034 | 0,9806 ± 0,0038 | 5,8 (3,2 / 2,7) | 1.574,3 ± 27,8 | 9,63 ± 0,30 | 31,98 ± 1,01 | 0,00062 ± 0,00001 | 0,102 ± 0,003 | 30,7 ± 0,9 |

Notas: erros FP/FN reconstruídos a partir de precisão e recall (150 phishing e 151 legítimos no teste).

**Avaliação por época, treino líquido e duração total**

| Modelo | T. Aval. por época (s) | Aval. (% do treino bruto) | T. Treino líquido (s) | TES (líquido) | Duração do processo (s)¹ |
|---|---|---|---|---|---|
| TinyBERT | 22,46 ± 0,34 | 5,5% | 384,95 ± 16,81 | 0,002585 ± 0,000111 | 430,30 ± 17,52 |
| DistilBERT | 166,93 ± 0,75 | 9,2% | 1.639,99 ± 5,01 | 0,000605 ± 0,000002 | 1.845,82 ± 4,69 |
| MobileBERT | 111,95 ± 0,98 | 7,1% | 1.469,49 ± 65,32 | 0,000672 ± 0,000030 | 1.614,72 ± 65,78 |
| ALBERT | 390,44 ± 3,32 | 12,3% | 2.779,52 ± 37,30 | 0,000354 ± 0,000005 | 3.229,80 ± 40,56 |
| MiniLM | 96,53 ± 1,42 | 6,1% | 1.477,81 ± 26,63 | 0,000664 ± 0,000012 | 1.607,17 ± 27,74 |

¹ Duração do processo inteiro no resumo do lote (carregamento dos dados e do modelo, pré-processamento, treino, teste e gravação).

### 8.7 Resultados por execução — todas as métricas

| Modelo | Exec. | Acc. | Prec. | Rec. | F1 | FP | FN | T. Treino bruto (s) | T. Aval. por época (s) | T. Treino líquido (s) | T. Teste (s) | Detecção (ms) | TES (bruto) | TES (líquido) | IES | RTDE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyBERT | 1 | 0,9967 | 0,9934 | 1,0000 | 0,9967 | 1 | 0 | 395,14 | 21,94 | 373,21 | 2,30 | 7,63 | 0,002522 | 0,002671 | 0,4339 | 130,60 |
| TinyBERT | 2 | 0,9934 | 1,0000 | 0,9867 | 0,9933 | 0 | 2 | 406,95 | 22,57 | 384,38 | 2,43 | 8,07 | 0,002441 | 0,002584 | 0,4090 | 123,12 |
| TinyBERT | 3 | 0,9934 | 0,9868 | 1,0000 | 0,9934 | 2 | 0 | 404,54 | 22,37 | 382,17 | 2,14 | 7,12 | 0,002456 | 0,002599 | 0,4633 | 139,44 |
| TinyBERT | 4 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 398,85 | 22,36 | 376,49 | 2,15 | 7,14 | 0,002491 | 0,002638 | 0,4622 | 139,12 |
| TinyBERT | 5 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 397,83 | 22,54 | 375,29 | 2,44 | 8,12 | 0,002497 | 0,002647 | 0,4065 | 122,35 |
| TinyBERT | 6 | 0,9900 | 0,9933 | 0,9867 | 0,9900 | 1 | 2 | 441,12 | 22,97 | 418,15 | 2,28 | 7,58 | 0,002244 | 0,002368 | 0,4340 | 130,63 |
| DistilBERT | 1 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 1.802,79 | 167,93 | 1.634,86 | 16,83 | 55,92 | 0,000551 | 0,000608 | 0,0590 | 17,77 |
| DistilBERT | 2 | 0,9934 | 0,9868 | 1,0000 | 0,9934 | 2 | 0 | 1.812,96 | 166,08 | 1.646,87 | 16,74 | 55,60 | 0,000548 | 0,000603 | 0,0594 | 17,86 |
| DistilBERT | 3 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 1.804,11 | 166,22 | 1.637,89 | 16,70 | 55,48 | 0,000551 | 0,000606 | 0,0595 | 17,91 |
| DistilBERT | 4 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 1.808,02 | 167,00 | 1.641,02 | 16,83 | 55,91 | 0,000549 | 0,000605 | 0,0590 | 17,77 |
| DistilBERT | 5 | 0,9934 | 1,0000 | 0,9867 | 0,9933 | 0 | 2 | 1.801,51 | 166,69 | 1.634,82 | 16,84 | 55,93 | 0,000551 | 0,000608 | 0,0590 | 17,76 |
| DistilBERT | 6 | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 2 | 1 | 1.812,10 | 167,65 | 1.644,45 | 16,80 | 55,80 | 0,000546 | 0,000602 | 0,0589 | 17,74 |
| MobileBERT | 1 | 0,9867 | 0,9932 | 0,9800 | 0,9866 | 1 | 3 | 1.515,13 | 111,50 | 1.403,63 | 11,13 | 36,97 | 0,000651 | 0,000703 | 0,0887 | 26,69 |
| MobileBERT | 2 | 0,9801 | 0,9800 | 0,9800 | 0,9800 | 3 | 3 | 1.528,11 | 110,91 | 1.417,20 | 11,06 | 36,75 | 0,000641 | 0,000692 | 0,0886 | 26,67 |
| MobileBERT | 3 | 0,9900 | 1,0000 | 0,9800 | 0,9899 | 0 | 3 | 1.529,23 | 112,37 | 1.416,86 | 11,07 | 36,78 | 0,000647 | 0,000699 | 0,0894 | 26,92 |
| MobileBERT | 4 | 0,9834 | 0,9866 | 0,9800 | 0,9833 | 2 | 3 | 1.636,59 | 111,19 | 1.525,40 | 11,08 | 36,82 | 0,000601 | 0,000645 | 0,0887 | 26,71 |
| MobileBERT | 5 | 0,9867 | 0,9932 | 0,9800 | 0,9866 | 1 | 3 | 1.668,78 | 112,11 | 1.556,66 | 11,17 | 37,10 | 0,000591 | 0,000634 | 0,0884 | 26,60 |
| MobileBERT | 6 | 0,9867 | 0,9932 | 0,9800 | 0,9866 | 1 | 3 | 1.610,76 | 113,60 | 1.497,15 | 11,13 | 36,98 | 0,000613 | 0,000659 | 0,0887 | 26,68 |
| ALBERT | 1 | 0,9801 | 0,9675 | 0,9933 | 0,9803 | 5 | 1 | 3.227,72 | 395,45 | 2.832,28 | 40,15 | 133,37 | 0,000304 | 0,000346 | 0,0244 | 7,35 |
| ALBERT | 2 | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 2 | 1 | 3.186,16 | 389,96 | 2.796,20 | 38,69 | 128,55 | 0,000311 | 0,000354 | 0,0256 | 7,70 |
| ALBERT | 3 | 0,9867 | 0,9740 | 1,0000 | 0,9868 | 4 | 0 | 3.178,30 | 390,39 | 2.787,91 | 39,02 | 129,65 | 0,000310 | 0,000354 | 0,0253 | 7,61 |
| ALBERT | 4 | 0,9801 | 0,9737 | 0,9867 | 0,9801 | 4 | 2 | 3.172,07 | 390,86 | 2.781,21 | 38,86 | 129,12 | 0,000309 | 0,000352 | 0,0252 | 7,59 |
| ALBERT | 5 | 0,9867 | 0,9867 | 0,9867 | 0,9867 | 2 | 2 | 3.106,58 | 385,02 | 2.721,56 | 38,99 | 129,53 | 0,000318 | 0,000363 | 0,0253 | 7,62 |
| ALBERT | 6 | 0,9801 | 0,9737 | 0,9867 | 0,9801 | 4 | 2 | 3.148,95 | 390,95 | 2.757,99 | 38,97 | 129,47 | 0,000311 | 0,000355 | 0,0251 | 7,57 |
| MiniLM | 1 | 0,9834 | 0,9801 | 0,9867 | 0,9834 | 3 | 2 | 1.560,46 | 96,55 | 1.463,90 | 9,44 | 31,35 | 0,000630 | 0,000672 | 0,1042 | 31,37 |
| MiniLM | 2 | 0,9834 | 0,9801 | 0,9867 | 0,9834 | 3 | 2 | 1.566,22 | 96,65 | 1.469,57 | 9,67 | 32,12 | 0,000628 | 0,000669 | 0,1017 | 30,61 |
| MiniLM | 3 | 0,9801 | 0,9800 | 0,9800 | 0,9800 | 3 | 3 | 1.630,60 | 98,94 | 1.531,66 | 10,22 | 33,95 | 0,000601 | 0,000640 | 0,0959 | 28,87 |
| MiniLM | 4 | 0,9734 | 0,9671 | 0,9800 | 0,9735 | 5 | 3 | 1.559,83 | 96,06 | 1.463,77 | 9,43 | 31,33 | 0,000624 | 0,000665 | 0,1032 | 31,07 |
| MiniLM | 5 | 0,9801 | 0,9800 | 0,9800 | 0,9800 | 3 | 3 | 1.569,42 | 96,46 | 1.472,96 | 9,55 | 31,72 | 0,000624 | 0,000665 | 0,1026 | 30,89 |
| MiniLM | 6 | 0,9834 | 0,9866 | 0,9800 | 0,9833 | 2 | 3 | 1.559,54 | 94,51 | 1.465,03 | 9,46 | 31,42 | 0,000631 | 0,000671 | 0,1040 | 31,30 |

Acurácia e erros por execução, em resumo:

| Modelo | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---|---|---|---|---|---|
| TinyBERT | 0,9967 (1) | 0,9934 (2) | 0,9934 (2) | 0,9934 (2) | 0,9934 (2) | 0,9900 (3) |
| DistilBERT | 0,9934 (2) | 0,9934 (2) | 0,9934 (2) | 0,9934 (2) | 0,9934 (2) | 0,9900 (3) |
| MobileBERT | 0,9867 (4) | 0,9801 (6) | 0,9900 (3) | 0,9834 (5) | 0,9867 (4) | 0,9867 (4) |
| ALBERT | 0,9801 (6) | 0,9900 (3) | 0,9867 (4) | 0,9801 (6) | 0,9867 (4) | 0,9801 (6) |
| MiniLM | 0,9834 (5) | 0,9834 (5) | 0,9801 (6) | 0,9734 (8) | 0,9801 (6) | 0,9834 (5) |

### 8.8 Comparação com o artigo

| Modelo | Acc. artigo | Acc. replicação | Δ (e-mails em 301) | T. treino rel. ao TinyBERT (artigo → replicação) | T. teste rel. ao TinyBERT (artigo → replicação) |
|---|---|---|---|---|---|
| TinyBERT | 0,9927 ± 0,0006 | 0,9934 ± 0,0021 | +0,2 | 1,00× → 1,00× | 1,00× → 1,00× |
| DistilBERT | 0,9934 ± 0,0005 | 0,9928 ± 0,0014 | −0,2 | 0,62× → 4,44× | 0,72× → 7,33× |
| MobileBERT | 0,9904 ± 0,0004 | 0,9856 ± 0,0034 | −1,4 | 1,22× → 3,88× | 1,57× → 4,85× |
| ALBERT | 0,9834 ± 0,0008 | 0,9839 ± 0,0044 | +0,2 | 1,31× → 7,78× | 1,46× → 17,07× |
| MiniLM | 0,9889 ± 0,0007 | 0,9806 ± 0,0039 | −2,5 | 0,57× → 3,86× | 1,00× → 4,20× |

### 8.9 Achados

1. **TinyBERT empata com o melhor modelo e domina em eficiência.** Diferença média para o DistilBERT: 0,2 e-mail. O DistilBERT custa 4,4× mais no treino e 7,3× mais na inferência. TinyBERT, DistilBERT e ALBERT ficam a menos de meio e-mail das médias do artigo.
2. **MobileBERT e MiniLM abaixo do artigo** (−1,4 e −2,5 e-mails). A ordenação muda em relação ao artigo (artigo: DistilBERT > TinyBERT > MobileBERT > MiniLM > ALBERT; aqui: TinyBERT ≈ DistilBERT > MobileBERT ≈ ALBERT > MiniLM). Entre MobileBERT, ALBERT e MiniLM as diferenças (1–1,5 e-mail) estão perto do erro-padrão binomial (≈ 1,5 e-mail); a separação entre o grupo superior (~2 erros) e o inferior (4–6 erros) é nítida.
3. **MobileBERT — erro sistemático.** Recall 0,9800 nas 6 execuções (3 FN, desvio zero): provavelmente os mesmos 3 phishing são sempre perdidos. Candidato prioritário para a análise de erros.
4. **ALBERT — sem platô em nenhuma das 6 execuções** (pendência da Seção 7.10; as 6 curvas foram conferidas). Em todas, a perda de treino cai já na 1ª época; a perda de avaliação final ficou entre ≈ 0,08 e 0,14. Persistem picos fortes na perda de treino (até ≈ 0,6–0,7), maiores que os dos demais modelos. A média (0,9839) coincide com o artigo e supera o piloto local de 02/10 (0,9701, preso no platô). Leitura: o platô é uma instabilidade que ocorre em parte das execuções, não um comportamento sistemático — 0 de 6 aqui, 1 de 1 no piloto local, 0 de 1 no Colab (Seção 7.8). Os erros do ALBERT são principalmente FP (3,5 em média, contra 1,3 FN).
5. **MiniLM — sobreajuste visível** (curva da execução 4 conferida). Perda de avaliação mínima na 2ª época (≈ 0,07), depois sobe e estabiliza em ≈ 0,16, com perda de treino ≈ 0. Com 10 épocas fixas e sem seleção (protocolo do artigo), o modelo paga o preço da superconfiança — mesmo mecanismo observado, em grau menor, no TinyBERT.
6. **Custo segue o tamanho das arquiteturas** (TinyBERT 3,9–7,8× mais rápido no treino e 4,2–17,1× na inferência), confirmando o piloto (Seção 6.4) e contrariando a Tabela 3 do artigo. A avaliação por época custa de 5,5% (TinyBERT) a 12,3% (ALBERT) do treino bruto; o TES líquido não altera a ordenação.
7. **Variação de tempo é da máquina, não do modelo.** O DistilBERT tem DP de 4,9 s no treino; o MobileBERT, 65,6 s (execuções 4–6 ≈ 110 s mais lentas), e a execução 6 do TinyBERT levou 441 s contra ≈ 400 s nas demais — com tempos de teste estáveis. Indica variação de carga da máquina durante o treino.

**Ressalva estatística.** As 6 execuções variam apenas a semente de treino, com o mesmo conjunto de teste: os desvios-padrão medem a **estabilidade do fine-tuning**, não a incerteza sobre dados novos. Para comparar modelos, vale o erro-padrão binomial (≈ ±0,5 p.p. com n = 301); diferenças de 1–2 e-mails não devem ser declaradas significativas.

### 8.10 Status

- **Este lote é a referência oficial da replicação desde 07/10** (Seções 1.3 e 5): usa o tamanho de amostra do artigo, os cinco modelos e 6 execuções por modelo, com dados idênticos em todas as execuções.
- Estudo detalhado dos resultados: iniciado em 08/10. Pendências consolidadas na Seção 9.5.

---

## 9. Conclusão e próximos passos

### 9.1 Síntese da trajetória

**Fase 2 (17/09–25/09) — corrigir os dados.** A replicação começou com resultados "bons demais" (precisão 1,0000), que levaram ao diagnóstico de vazamento de dados (duplicatas entre treino e teste). A correção passou por uma solução desenhada mas nunca executada (23/09) e pela descoberta de dois bugs reais de implementação (código morto no Módulo A e rótulos invertidos, 24/09). O lote 25/09(b) foi o primeiro com pipeline verificado linha a linha e sobreposição treino/teste comprovadamente nula (Seções 1.2, 3 e 4).

**Fase 3 (01/10–07/10) — corrigir o método e ampliar a comparação.** (1) Remoção do viés de seleção de modelo no teste, com protocolo fiel ao artigo (01/10, Seção 3.4); (2) correção da tabela de referência do artigo (Seção 2); (3) expansão para os cinco modelos do artigo (02/10, Seção 6); (4) reprodução em outro ambiente e correção da dependência da ordem de leitura dos arquivos (Colab, 03–04/10, Seção 7); (5) automação da execução com registro de ambiente e impressões digitais (04/10, Seção 8.2); (6) alinhamento do undersampling ao artigo, 502/501 (07/10, Seção 8.4); (7) lote final com 5 modelos × 6 execuções (07/10, Seções 8.5–8.10).

### 9.2 Resultado de referência atual

Lote final de 07/10 (Seção 8): TinyBERT (PHILDER) com Acurácia 0,9934 ± 0,0021 e F1 0,9933 ± 0,0021, contra 0,9927 ± 0,0006 no artigo — o desempenho preditivo do PHILDER foi reproduzido. O TinyBERT empata com o DistilBERT (0,9928 ± 0,0014), o modelo mais preciso, e é o mais eficiente em todas as medidas: 3,9–7,8× mais rápido no treino e 4,2–17,1× na inferência que os demais modelos, no mesmo hardware. A afirmação central do artigo — o TinyBERT oferece o melhor equilíbrio entre acurácia e custo — se sustenta, e com margem maior que a sugerida pela Tabela 3 do próprio artigo. As diferenças em relação ao artigo (MiniLM e MobileBERT abaixo; tempos de treino maiores e de inferência menores) estão registradas na Seção 5.3.

### 9.3 Lições metodológicas

1. **Plausibilidade das métricas não substitui verificação direta** do código executado e dos dados resultantes (vazamento de dados e código morto, Seção 1.2).
2. **Usar o teste para decidir o treino infla o resultado,** mesmo sem nenhum e-mail do teste nos pesos (viés de seleção, ≈ 0,4 p.p.; Seção 3.4).
3. **Contagens iguais não provam conteúdo igual;** sementes fixas não garantem reprodutibilidade se a ordem de entrada dos dados depender do sistema operacional (impressão digital MD5, Seção 7.2).
4. **Uma gravação "concluída sem erro" não prova que o conteúdo certo chegou ao destino** — só a leitura do arquivo resultante prova (Seção 6.2).
5. **Repetir execuções com os mesmos dados mede a estabilidade do treino, não a incerteza do resultado** (pseudo-replicação, Seções 3.4 e 8.9); com n = 301, diferenças de 1–2 e-mails não são distinguíveis (erro-padrão ≈ ±0,5 p.p.).
6. **Métricas de eficiência só são comparáveis no mesmo hardware,** que deve ser reportado junto com elas; a vantagem do modelo menor cresce em hardware mais fraco (Seções 7.5 e 7.9).
7. **Instabilidade de fine-tuning precisa de várias execuções para aparecer** (platô do ALBERT: 1 de 1 no piloto, 0 de 6 no lote final; Seções 7.8 e 8.9).

### 9.4 Registro histórico — conclusão de 25/09

Conclusão registrada ao fim da Fase 2, mantida para registro. O resultado de referência nela citado (25/09(b)) foi substituído (Seção 1.3).

O resultado de 25/09(b) é adotado como referência definitiva da replicação: metodologicamente correto (sem vazamento de dados, confirmado via checagem programática explícita de sobreposição treino/teste, não apenas pela plausibilidade das métricas), com ambos os pools do dataset (legítimos/SpamAssassin e phishing/Nazario) deduplicados antes da amostragem, e com o pipeline de execução do notebook verificado linha a código — o primeiro da replicação a atingir esse nível de certeza. O lote 25/09(a), mantido lado a lado na Seção 3, documenta o passo intermediário (dedup só do phishing) e permite isolar o efeito específico de estender a deduplicação também ao lado dos legítimos (Seção 4). A evolução registrada nesta linha do tempo — da detecção inicial de um resultado "bom demais" (21/09), passando por uma correção desenhada mas nunca de fato executada (23/09), até a descoberta dos dois bugs reais de implementação e a comparação entre as duas técnicas de deduplicação validadas em 25/09(a) e (b) — é, em si, material especialmente relevante para a entrega de "análise crítica dos métodos de aferição de desempenho e eficácia" prevista na Fase 2: a lição central é que plausibilidade de métricas não substitui verificação direta do código executado e dos dados resultantes.

Itens em aberto para próximas atualizações deste documento:
- Repetir a execução de referência (25/09(b)) com sementes diferentes de `random_state` (todas as 6 execuções atuais usam a mesma semente/configuração, variando apenas por não-determinismo de treino), para isolar variância de semente de variância de treino.
- Incorporar novos marcos à linha do tempo conforme forem executados.
- A extensão da deduplicação prévia ao lado do SpamAssassin (item que estava em aberto neste documento até a atualização anterior) já foi implementada e validada como parte da correção de 24–25/09 — item concluído, com os dois lotes (a) e (b) documentando o antes/depois dessa extensão.

> **Situação em 08/10:** o primeiro item (sementes) teve decisão — manter o split e as sementes fixos no protocolo principal e avaliar a variação de sementes no estudo de ablação (Seção 9.5). O segundo é atendido continuamente por este documento. O terceiro já estava concluído.

### 9.5 Pendências consolidadas (08/10/2026)

**Estudo dos resultados**
- Estudo detalhado do lote final (iniciado em 08/10).
- Análise de erros: os 3 phishing que o MobileBERT perde em todas as execuções (FN constante); os FP do ALBERT e do MiniLM; os e-mails errados por todas as execuções do TinyBERT em 01/10 (Seção 3.4, item 4).
- Conferir as curvas de perda de TinyBERT, MiniLM, DistilBERT e MobileBERT no lote final (só o ALBERT foi conferido nas 6 execuções; MiniLM em 1).
- Investigar o padrão "treino mais lento, inferência mais rápida" em relação ao artigo (comprimento de sequência, batch de inferência, tokenização dentro do tempo de teste; Seções 3.4 e 5.3).

**Experimentos complementares (separados do protocolo principal)**
- Estudo de ablação com variação de sementes (`random_state` do split e `seed` do treino).
- Warmup / learning rate menor para o ALBERT, sem usar o teste para escolher a configuração do protocolo principal (Seção 7.10).
- Experimento GPU vs. CPU, para testar a hipótese de compressão das diferenças de tempo em hardware potente (Seções 6.4 e 7.9).
- Comparação 501/501 vs. 502/501 em várias execuções, se for útil separar o efeito do dado do efeito do acaso.

**Código e reprodutibilidade (cada alteração de código depende de consentimento explícito)**
- Commit/push para o GitHub do estado validado (`rodar_lote.py`, `database_consolidation.py`, `database_balance.py`, notebook e documentos).
- `requirements.txt` com versões fixas (Seção 7.6).
- Curva de perda em escala logarítmica ou com recorte dos passos iniciais (MobileBERT ilegível em escala linear; Seção 6.5).
- O cabeçalho do `rodar_lote.py` cita "Analise_dos_Resultados.md, Seção 8.2", que após a reorganização de 08/10 passou a ser a Seção 7.2.

**Organização de arquivos e documentos**
- Mover `resultados/tinybert/resultado_final-7_07102026.csv` (cópia do lote de validação 501/501) para `resultados/tinybert/teste e validacao/`; decidir se o CSV da validação 502/501 (Seção 8.4) deve ser recuperado de `.ipynb_checkpoints`.
- Atualizar `Embasamento_Teorico_Metricas_PHILDER.md` (versão de 25/09) e o handoff do projeto com a referência atual.
- Remover cópias duplicadas deste documento no Projeto.
- Iniciar a redação do artigo (Overleaf, formato IEEE) a partir das Seções 5, 8 e 9.
