# Análise dos Resultados — Replicação PHILDER
### Aplicações Distribuídas — PPGEE/UnB | Fases 2 e 3

> Este documento registra, em ordem cronológica, cada execução do pipeline de replicação do PHILDER (BISPO, G. D. et al., IEEE Access, 2025) e compara os resultados obtidos com os do artigo original. Para a fundamentação teórica das métricas, a análise crítica da normalização de TES/RTDE na Tabela 4 do artigo e o detalhamento estatístico completo do achado de vazamento de dados (Seções 3.5, 5.5 e 5.6), ver `Embasamento_Teorico_Metricas_PHILDER.md`.

> **Reorganização de 08/10/2026.** A conclusão foi movida para o final do documento (Seção 10) e as antigas Seções 7, 8 e 9 passaram a ser 6, 7 e 8 (referências cruzadas atualizadas). O resultado de referência oficial passou a ser o **lote final de 07/10/2026 — 5 modelos × 6 execuções, undersampling 502/501** (Seções 5 e 8). Os registros anteriores foram preservados; quando um trecho ficou superado, isso é indicado no próprio trecho.

> **Atualização de 09/10/2026.** Nova Seção 9 (execuções complementares de 09/10: rodadas extras de MiniLM, ALBERT e MobileBERT, experimento do ALBERT com os hiperparâmetros do artigo original e preparação da execução no Kaggle). A conclusão passou a ser a Seção 10.

> **Atualização de 01/11/2026.** Nova Seção 1.4: evidência do processo de deduplicação (exemplos de e-mails brutos diferentes que se tornam iguais após o pré-processamento, com o código da célula usada) e observações sobre o pré-processamento. Nova Seção 1.5: divergência do Módulo B em relação ao artigo (endereços de e-mail não eram mascarados), quantificação do artefato `jose@monkey.org`, correção do código, novos hashes de referência e plano do novo lote.

> **Atualização de 02/11/2026.** Nova Seção 10: **lote oficial com máscaras de links e e-mails** (5 modelos × 6 execuções, 01–02/11/2026). Por decisão do usuário, **passa a ser o resultado de referência oficial da replicação**; o lote de 07/10 (Seções 5 e 8) fica como registro histórico e como condição "sem máscara" da ablação (Seção 10.4). A conclusão passou a ser a Seção 11 (referências cruzadas atualizadas).

**Estrutura do documento**

| Seção | Conteúdo |
|---|---|
| 1 | Linha do tempo de todas as ações e medições; notas sobre o tamanho do teste e sobre as trocas de resultado de referência |
| 2 | Resultados do artigo original (valores de referência e inconsistências internas) |
| 3 | TinyBERT (PHILDER) por execução — Fase 2 (17/09–25/09) e protocolo fiel ao artigo (01/10) |
| 4 | Análise evolutiva — o que mudou e por quê, de 17/09 a 07/10 |
| 5 | Comparação com o artigo do lote de 07/10 (referência até 01/11; a atual está na Seção 10.5) |
| 6 | Expansão para os 5 modelos — piloto de 02/10 |
| 7 | Reprodutibilidade entre ambientes — Google Colab (03–04/10) |
| 8 | Automação, alinhamento do undersampling e lote final (04–07/10) |
| 9 | Execuções complementares de 09/10: rodadas extras (MiniLM, ALBERT, MobileBERT), experimento de hiperparâmetros do ALBERT, execução no Kaggle |
| 10 | **Lote oficial com máscaras de links e e-mails (01–02/11) — resultado de referência atual**; ablação com/sem máscara; comparação com o artigo |
| 11 | Conclusão, lições metodológicas, hiperparâmetros recomendados pelos autores e pendências consolidadas |

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
| 08/10/2026 | — | Reorganização deste documento: linha do tempo completada, nota de estratificação corrigida (Seção 1.1), histórico de referências (Seção 1.3), comparação final atualizada (Seção 5), conclusão movida para o final (Seção 10) |
| 09/10/2026 | — | **Levantamento dos hiperparâmetros de fine-tuning recomendados** nos model cards do Hugging Face e nos artigos originais dos cinco modelos. Os model cards não trazem receita de fine-tuning; as recomendações estão nos artigos. O protocolo único deste trabalho fica fora das faixas recomendadas para ALBERT e MiniLM. Ver Seção 11.6 |
| 09/10/2026 (07h47–13h22) | Mesmos dados (502/501), protocolo "artigo", `rodar_lote.py`, PC local | **Rodadas complementares:** MiniLM × 3 (07h47–09h10), ALBERT × 3 e MobileBERT × 3 (09h19–13h22; a 1ª tentativa do ALBERT foi interrompida manualmente às 09h18, sem resultado). Todas com os hashes de referência. MobileBERT repete exatamente 07/10 (0,9856); ALBERT (0,9790) e MiniLM (0,9790) empatam, um pouco abaixo de 07/10. Ver Seção 9.2 |
| 09/10/2026 (tarde) | Notebook do Kaggle (`philder_kaggle.ipynb`, fora do repositório) | **Preparação da execução no Kaggle:** `drive.mount` do Colab não funciona no Kaggle; criado notebook próprio com três fontes de código (GitHub fixado em commit, zip no Drive, Kaggle Dataset), conferência de MD5 dos dados brutos e processados e registro de proveniência. Execução bloqueada: a ativação da internet exige verificação de telefone, e o código SMS não chegou. Ver Seção 9.4 |
| 09/10/2026 (20h48–21h38) | ALBERT, 502/501, **lr 1e-5, batch 32, warmup 6%** (Tabela 14 do artigo do ALBERT), demais parâmetros iguais; célula independente no `philder.ipynb` | **Experimento complementar de hiperparâmetros (1 rodada):** treino estável, sem platô e sem picos de perda; acurácia 0,9734 (6 FP + 2 FN), sem ganho em relação ao protocolo principal. Ver Seção 9.3 |
| 01/11/2026 | Células temporárias de inspeção no `philder.ipynb` (só leitura; não alteram o pipeline) | **Evidência da deduplicação:** todas as 80 duplicatas descartadas (93 e-mails de phishing em 49 grupos; 55 de ham em 31 grupos) só se tornam iguais **depois** do pré-processamento — nenhuma tem texto bruto idêntico. Observações sobre o pré-processamento: remoção de negações pelas stopwords, quebra de palavras acentuadas, `\r` no Nazario, conteúdo repetido dentro do mesmo e-mail e quase-duplicatas de campanhas. Ver Seção 1.4 |
| 01/11/2026 | `philder/data_processing.py`: novas funções `substituir_emails` e `desfazer_quebras_quoted_printable`, regex de URL ampliada e parâmetro `mascarar_links_e_emails=True` em `pipeline_processamento` (aprovado pelo usuário) | **Divergência corrigida:** o Módulo B não mascarava endereços de e-mail, ao contrário do artigo (Seção III-B, "URL and Email Address Removal"). O endereço da caixa-isca do Nazario virava "jose monkey org" em ≈ 40% dos phishing e < 1% dos legítimos. Com a máscara: "monkey" cai para 12,2% dos phishing; deduplicação do phishing 1.565 → 1.471 (antes 1.472); **novos hashes** (balanceado `ba57e8a3…`). O lote de 07/10 passa a ser a condição "sem máscara". Ver Seção 1.5 |
| 01–02/11/2026 | Rodada de validação do TinyBERT (arquivos `valida_*`) e lote `python rodar_lote.py --modelos todos --rodadas 6` com as máscaras (19h58 de 01/11 às 09h47 de 02/11) | **Novo resultado de referência oficial.** 30 de 30 execuções "ok", hashes idênticos. TinyBERT 0,9939 ± 0,0039 (artigo 0,9927 ± 0,0006); DistilBERT 0,9906; MobileBERT 0,9906; MiniLM 0,9884; ALBERT 0,9878. Os cinco modelos ficam a ≤ 1,3 e-mail das médias do artigo; FN caem em todos os modelos. Ver Seção 10 |

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
| 07/10–01/11 | Lote final de 07/10 — 5 modelos × 6 execuções, protocolo fiel ao artigo, undersampling 502/501, `rodar_lote.py`, PC local (Seções 5 e 8) | O Módulo B não mascarava endereços de e-mail, ao contrário do artigo (Seção 1.5). Passa a ser a condição "sem máscara" da ablação (Seção 10.4) |
| **02/11 em diante** | **Lote oficial de 01–02/11 — mesma configuração de 07/10, com máscaras de links e e-mails (Módulo B fiel à Seção III-B do artigo)** | — (referência atual; Seção 10) |

Os resultados substituídos continuam registrados nas Seções 3, 6 e 7: cada um documenta uma etapa da correção do método e serve de comparação para isolar o efeito de cada mudança.


### 1.4 Evidência do processo de deduplicação (01/11/2026)

**Objetivo.** Mostrar, com exemplos reais dos corpora, como funciona a deduplicação prévia adotada desde 25/09 (Seção 1.2): os e-mails são comparados pelo **texto já processado** (`body_processado`, Módulo B), e não pelo texto bruto. Para isso foram usadas duas células temporárias no `philder.ipynb`, que apenas leem e exibem dados — não alteram o pipeline, os módulos do pacote `philder` nem os resultados.

**Critério reproduzido.** O mesmo de `_amostrar_sem_duplicatas` (chamada por `amostrar_quantidade_artigo(..., deduplicar_antes=True)`, Módulo A): para cada e-mail bruto, `extrair_campos_email` separa cabeçalho e corpo; `pipeline_processamento` gera o `body_processado` (remoção de cabeçalho residual, troca de URLs por "url", minúsculas, remoção de caracteres fora de `a-z0-9`, remoção de stopwords em inglês); dois e-mails são duplicatas quando o `body_processado` é idêntico, e só o primeiro de cada grupo permanece no pool (`duplicated(keep="first")`).

#### 1.4.1 Resultado

| Corpus | E-mails brutos | Únicos após a deduplicação | Descartados | Grupos de duplicatas | Grupos com texto **bruto** idêntico | Grupos iguais **só após** o processamento |
|---|---|---|---|---|---|---|
| Phishing (Nazario) | 1.565 | 1.472 | 93 | 49 | **0** | 49 |
| Ham (SpamAssassin, `easy_ham`) | 2.501 | 2.446 | 55 | 31 | **0** | 31 |

**Achado principal: nenhuma das 80 duplicatas teria sido detectada comparando o texto bruto.** Todas só se tornam iguais depois do pré-processamento. Isso justifica, com números, a decisão de deduplicar pelo `body_processado` — o mesmo texto que o modelo recebe e o mesmo critério usado na checagem de sobreposição treino/teste (Seção 3.1, lote 25/09(b)).

#### 1.4.2 Exemplos

**Phishing (Nazario): posições 185 (mantido) e 314 (descartado).** Os dois são o golpe "Your email (…) is almost full / Upgrade Now", com 2.774 e 2.776 caracteres. Os textos brutos diferem em detalhes mínimos: "kindly **t**o add" contra "kindly **d**o add" (duas ocorrências, porque o texto se repete dentro do e-mail) e dois espaços a mais antes de uma linha de asteriscos. Como "to" e "do" estão na lista de stopwords, os dois `body_processado` ficam idênticos (1.527 caracteres).

**Ham (SpamAssassin): posições 454 (mantido) e 460 (descartado).** É a **mesma mensagem da lista FoRK entregue duas vezes** (9.650 caracteres cada). O corpo é igual, mas o cabeçalho SMTP difere em 44 trechos: horário de recebimento (11:42:31 contra 11:42:49), identificadores de servidor (`42C4F16F77` contra `C62F16F21`) etc. Como o Módulo A separa o cabeçalho do corpo, os dois `body_processado` ficam idênticos (4.505 caracteres).

Os dois exemplos ilustram as duas origens de duplicatas: no phishing, **variações mínimas de uma mesma campanha** que o pré-processamento neutraliza; no ham, **a mesma mensagem recebida mais de uma vez**, distinta apenas no cabeçalho.

#### 1.4.3 Código da célula (documentado)

A célula deve ser colada **depois da célula do Módulo A**, porque usa as listas `brutos_legitimos` e `brutos_phishing` e os módulos `consolidacao` e `processamento` já importados.

```python
# ===== CÉLULA TEMPORÁRIA (01/11/2026) — evidência da deduplicação pelo texto processado =====
# Mostra, para cada rótulo, 2 e-mails BRUTOS diferentes que viram o MESMO body_processado
# após o Módulo B — exatamente o critério usado em amostrar_quantidade_artigo(deduplicar_antes=True).
# Usa brutos_legitimos / brutos_phishing já carregados na célula do Módulo A. Apenas exibe; pode ser apagada.
import difflib
import pandas as pd

LIMITE = 600   # caracteres exibidos de cada texto

def _grupos_duplicados(emails_brutos):
    campos = [consolidacao.extrair_campos_email(m) for m in emails_brutos]
    df_tmp = processamento.pipeline_processamento(pd.DataFrame(campos))
    df_tmp["bruto"] = emails_brutos
    dup = df_tmp[df_tmp.duplicated(subset="body_processado", keep=False)]
    grupos = list(dup.groupby("body_processado", sort=False))
    # separa grupos em que os brutos são idênticos dos que só coincidem DEPOIS do processamento
    so_apos = [(t, g) for t, g in grupos if g["bruto"].nunique() > 1]
    identicos = [(t, g) for t, g in grupos if g["bruto"].nunique() == 1]
    return df_tmp, so_apos, identicos

def _mostrar(texto):
    texto = texto.replace("\r", "\n")
    return texto[:LIMITE] + (f" [...] (+{len(texto) - LIMITE} caracteres)" if len(texto) > LIMITE else "")

def _onde_diferem(a, b, contexto=60):
    a, b = a.replace("\r", "\n"), b.replace("\r", "\n")
    blocos = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    for tag, i1, i2, j1, j2 in blocos[:3]:
        print(f"  • {tag}: A[{i1}:{i2}] = {a[i1:i2][:contexto]!r}  |  B[{j1}:{j2}] = {b[j1:j2][:contexto]!r}")
    if len(blocos) > 3:
        print(f"  • ... e mais {len(blocos) - 3} trecho(s) diferente(s)")

def evidenciar(emails_brutos, nome):
    df_tmp, so_apos, identicos = _grupos_duplicados(emails_brutos)
    removidos = df_tmp["body_processado"].duplicated(keep="first").sum()
    print("#" * 95)
    print(f"{nome}: {len(emails_brutos)} e-mails brutos -> {len(emails_brutos) - removidos} únicos ({removidos} descartados)")
    print(f"  grupos de duplicatas: {len(so_apos) + len(identicos)} "
          f"({len(identicos)} com texto bruto idêntico; {len(so_apos)} só iguais APÓS o processamento)")
    print("#" * 95)
    if not so_apos:
        print("Nenhum par diferente no bruto e igual após o processamento.\n")
        return
    texto_proc, g = max(so_apos, key=lambda tg: len(tg[0]))   # o par com mais conteúdo, para ficar ilustrativo
    a, b = g["bruto"].iloc[0], g["bruto"].iloc[1]
    print(f"\n--- E-mail A (bruto, posição {g.index[0]} na lista, {len(a)} caracteres) ---\n{_mostrar(a)}")
    print(f"\n--- E-mail B (bruto, posição {g.index[1]} na lista, {len(b)} caracteres) ---\n{_mostrar(b)}")
    print("\n--- Onde os brutos diferem ---")
    _onde_diferem(a, b)
    print(f"\n--- body_processado (IGUAL nos dois; {len(texto_proc)} caracteres) ---\n{_mostrar(texto_proc)}")
    print(f"\nResultado: só o 1º ({g.index[0]}) fica no pool; o(s) outro(s) {list(g.index[1:])} é(são) descartado(s).\n")

evidenciar(brutos_phishing, "PHISHING (Nazario)")
evidenciar(brutos_legitimos, "HAM / LEGÍTIMO (SpamAssassin)")
```

**Como funciona, função a função:**

- **`_grupos_duplicados(emails_brutos)`** — reproduz o critério da deduplicação do projeto: aplica `extrair_campos_email` a cada e-mail, roda `pipeline_processamento` e guarda o texto bruto ao lado (`df_tmp["bruto"]`). Seleciona todos os registros cujo `body_processado` se repete (`duplicated(keep=False)`, que marca **todos** os membros de cada grupo, não só os descartados) e os agrupa pelo texto processado. Depois separa os grupos em duas listas: `identicos` (o texto bruto também é igual — seriam pegos por uma deduplicação ingênua) e `so_apos` (o bruto difere e só o texto processado coincide). O índice de `df_tmp` é a posição do e-mail na lista bruta, o mesmo mapeamento usado em `_amostrar_sem_duplicatas`.
- **`_mostrar(texto)`** — prepara o texto para exibição: troca `\r` por `\n` (ver Seção 1.4.4, item 1) e corta em `LIMITE` caracteres, informando quantos foram omitidos.
- **`_onde_diferem(a, b)`** — usa `difflib.SequenceMatcher` (com `autojunk=False`, para não ignorar caracteres frequentes) para listar os trechos em que os dois e-mails brutos diferem (`replace`, `insert`, `delete`), mostrando até 3 trechos.
- **`evidenciar(emails_brutos, nome)`** — imprime as contagens (brutos, únicos, descartados, grupos e quantos são idênticos no bruto) e escolhe, entre os grupos `so_apos`, o de **texto processado mais longo** (critério só para o exemplo ficar ilustrativo, sem efeito sobre os dados). Mostra os dois primeiros e-mails brutos do grupo, onde diferem, o `body_processado` comum e qual posição é mantida (a primeira) e quais são descartadas.

**Validação.** A célula foi executada fora do notebook sobre os mesmos arquivos de dados (`Nazario.csv` e `20030228_easy_ham.tar.bz2`) e reproduziu exatamente as contagens que o Módulo A imprime no log de cada execução (1.565 → 1.472 e 2.501 → 2.446).

#### 1.4.4 Observações sobre o pré-processamento

Uma primeira célula temporária, que mostra um e-mail de cada classe do `df` já processado (`body` original ao lado do `body_processado`), revelou cinco pontos, conferidos diretamente no `Nazario.csv`:

1. **`\r` no Nazario.** Parte dos e-mails usa só `\r` (retorno de carro) como quebra de linha — o e-mail da posição 4 do `df`, por exemplo, tem 120 `\r` e nenhum `\n`. No Jupyter, cada `\r` faz o texto seguinte sobrescrever o início da linha, o que embaralha a **exibição** (não os dados). O `body_processado` não é afetado, porque o Módulo B colapsa todos os espaços em branco. Nas células de inspeção, o texto é exibido com `.replace("\r", "\n")`.
2. **Conteúdo repetido dentro do mesmo e-mail.** Alguns e-mails do Nazario trazem a mensagem duas vezes no próprio corpo (provavelmente as versões texto e HTML coladas no CSV). Não é falha da deduplicação, que compara e-mails entre si. O efeito no modelo é pequeno: com `max_length=128`, só os primeiros ≈ 128 tokens são lidos.
3. **A remoção de stopwords apaga negações.** "Please **do not** reply" vira "please reply", porque "do" e "not" estão na lista do NLTK. Para modelos da família BERT, que interpretam a frase em contexto, isso remove informação. O Módulo B segue o pré-processamento descrito no artigo e **não foi alterado**; o ponto entra como limitação e como candidato ao estudo de ablação (com e sem remoção de stopwords). *Complemento (01/11):* o artigo descreve a etapa como remoção de palavras comuns, citando como exemplo **pronomes e conjunções** (Seção III-B); não diz qual lista usa nem menciona negações. A lista do NLTK vai além desses exemplos (inclui "not", "no", "nor" e auxiliares). É uma escolha de implementação nossa, não uma exigência do artigo, e fica registrada como possível divergência.
4. **Letras acentuadas quebram palavras.** O filtro `[^a-z0-9\s]` troca letras acentuadas por espaço: "pièces" vira "pi ces", "immédiatement" vira "imm diatement". O impacto é pequeno num corpus quase todo em inglês.
5. **Quase-duplicatas de campanhas.** O Nazario contém e-mails da mesma campanha com textos quase iguais, mas não idênticos (ex.: dois e-mails "HRM / New Payroll Bulletin", com 854 e 2.895 caracteres). A deduplicação exata não os remove. Se um cair no treino e outro no teste, o modelo pode "reconhecer" a campanha — uma forma mais sutil do vazamento corrigido em setembro. **Não verificado ainda**; registrado como pendência (checagem de similaridade treino/teste, Seção 11.5).

### 1.5 Divergência no Módulo B: endereços de e-mail não eram mascarados (01/11/2026)

#### 1.5.1 Como foi encontrada

Na saída da célula da Seção 1.4, o usuário notou que o endereço `jose@monkey.org` aparecia no texto processado como "jose monkey org": só o "@" e o "." eram trocados por espaço (pelo filtro `[^a-z0-9\s]` de `normalizar_texto`). O artigo descreve outra coisa. Na Seção III-B ("URL and Email Address Removal"), **URLs e endereços de e-mail** são substituídos por palavras genéricas, para que o modelo não aprenda padrões específicos e generalize a detecção. O Módulo B implementava só a parte das URLs (`substituir_urls`). Era, portanto, uma **divergência da replicação**, presente em todos os resultados até 01/11 — inclusive no lote de referência de 07/10.

#### 1.5.2 Por que importa: um atalho para o modelo

`jose@monkey.org` é o endereço da caixa-isca (*honeypot*) onde o corpus Nazario foi coletado. Ele aparece no corpo de muitos phishing (saudação, cópia do destinatário, cabeçalhos colados) e quase nunca nos legítimos do SpamAssassin. Medido no conjunto balanceado de referência (502/501):

| Palavra no `body_processado` | Phishing (sem máscara) | Phishing (com máscara) | Legítimos (sem → com) |
|---|---|---|---|
| "monkey" | 40,3% | 12,2% | 0,8% → 0,6% |
| "jose" | 41,9% | 11,2% | 1,0% → 1,0% |
| placeholder "EMAILADDRESS" | — | 47,1% | — → 57,2% |

No texto bruto, 185 dos 501 phishing contêm literalmente `jose@monkey.org`.

Uma palavra presente em ≈ 40% de uma classe e em < 1% da outra é uma **correlação espúria**: o modelo pode acertar por reconhecer de onde o e-mail veio, sem aprender o que torna uma mensagem phishing. É o *shortcut learning* descrito por Arp et al. [41] ("spurious correlations") — exatamente o risco que o artigo quer evitar com a máscara. Em termos do livro do Prince (*Understanding Deep Learning*, Cap. 8, "Measuring performance"), o teste continua vindo da mesma distribuição do treino, então o atalho **não aparece como erro no teste**: a acurácia pode ficar alta e mesmo assim não generalizar para e-mails de outras fontes. Isso é coerente com o alerta do projeto sobre resultados "bons demais".

#### 1.5.3 Alteração no código (aprovada pelo usuário em 01/11/2026)

**Critério definido pelo usuário:** todo link e todo endereço de e-mail do corpo devem ser substituídos. Nomes e outros identificadores (ex.: "Dear jose") e domínios soltos sem `http`/`www` (ex.: "sf.net", "apple.com", que podem ser outras referências) fazem parte da mensagem e **ficam no texto**.

A alteração foi feita em duas etapas no mesmo dia:

- **1ª versão** (md5 `f0b438b0…`): só a máscara de endereços de e-mail.
- **Versão final** (md5 `4c8eabe5…`): uma varredura dos dois corpora atrás de links e e-mails que sobravam depois das máscaras levou aos ajustes da tabela abaixo, aprovados pelo usuário (itens 1–4).

| Caso que escapava | Exemplo real do corpus | Ajuste |
|---|---|---|
| Maiúsculas | `WWW.USAA.COM` | regex de URL com `re.IGNORECASE` |
| FTP | `ftp://ftp.ximian.com/pub/…` | `ftp` incluído |
| Barra única / dois-pontos duplicados | `http:/apple.com/…`, `http:://intranet.monkey.org` | `:[:=\s]*/{1,2}` |
| Quebra *quoted-printable* ("=" no fim da linha) | `https:=⏎//u1366302.ct.sendgrid…`; no CSV, às vezes já achatada em `https:= //…` | nova função `desfazer_quebras_quoted_printable` (antes das máscaras) e tolerância a `= ` na regex |
| Resto do `mailto:` | `[mailto: EMAILADDRESS ]` (207 legítimos, 1 phishing) | `mailto:` absorvido pela máscara |
| Domínio sem ponto / truncado | `125117@localhost`, `dmcmann@b...`, `Sale@....` | domínio sem ponto e com reticências aceito |
| Outras formas | `x@[74.208.166.58]`, `bounces+whisper=oz.net@python.org`, `jose@= monkey.org`, `RBL:…:@paypal.com:` | IP entre colchetes, "=" na parte local, quebra achatada após o "@", domínio sem parte local (só com ponto, para não pegar "@home") |

Código final (trechos):

```python
_REGEX_URL_ATE_01112026 = re.compile(r"https?://\S+|www\.\S+")          # comportamento antigo
_REGEX_URL = re.compile(r"(?:https?|ftp):[:=\s]*/{1,2}\S*|www\.\S+", re.IGNORECASE)
_REGEX_EMAIL = re.compile(
    r"(?:mailto:\s*)?"
    r"(?:[\w.+=-]+@(?:=\s*)?(?:\[[\d.]+\]|\.{3,}|[\w-]+(?:\.[\w-]+)*(?:\.{3})?)"
    r"|@[\w-]+(?:\.[\w-]+)+)",
    re.IGNORECASE,
)
_REGEX_QUEBRA_QP = re.compile(r"=(?:\r\n|\r|\n)")

def pipeline_processamento(df, mascarar_links_e_emails: bool = True):
    ...
        texto = remover_cabecalho_smtp(linha["body"], linha["smtp_headers"])
        if mascarar_links_e_emails:  # 01/11/2026
            texto = desfazer_quebras_quoted_printable(texto)
            texto = substituir_urls(texto)      # -> " URL "
            texto = substituir_emails(texto)    # -> " EMAILADDRESS "
        else:  # comportamento até 01/11/2026
            texto = _REGEX_URL_ATE_01112026.sub(" URL ", texto)
        texto = normalizar_texto(texto)
        texto = remover_stopwords(texto)
```

- **Ordem:** as quebras *quoted-printable* são desfeitas primeiro, para reconstituir links e endereços partidos. As URLs são trocadas antes dos e-mails, para não quebrar URLs que contêm "@". Tudo acontece antes de `normalizar_texto`, que destruiria o "@" e os "://".
- **Placeholder:** "EMAILADDRESS", diferente da palavra comum "email", que é frequente no próprio texto das mensagens. Depois da normalização, ele vira "emailaddress".
- **Compatibilidade:** `mascarar_links_e_emails=False` reproduz exatamente o comportamento anterior. Isso foi conferido: com `False`, o pipeline gera os hashes de 07/10 (`b5eed2f9…`, `d7ee4165…`, `c2aecb1b…`).
- **Verificação de cobertura:** depois das máscaras, a varredura dos 1.565 phishing e dos 2.501 legítimos não encontrou nenhum link `http`/`https`/`ftp`/`www` nem endereço de e-mail. Os "@" que sobram não são endereços: CSS (`/*@editable*/`), código Perl (`$@`, `@ARGV`), interjeições ("LOL@#!") e endereços que já vieram anonimizados com asteriscos.
- Os módulos A e C, o `rodar_lote.py` e o notebook **não foram alterados**.

#### 1.5.4 Efeito nos dados

| Item | Sem máscara (até 01/11) | Com máscaras (a partir de 01/11) |
|---|---|---|
| Deduplicação do phishing | 1.565 → 1.472 (93 descartadas) | 1.565 → 1.471 (94 descartadas) |
| Deduplicação dos legítimos | 2.501 → 2.446 (55) | 2.501 → 2.446 (55) |
| Balanceado (502 / 501) | `b5eed2f9bf4f8dd5bd32fe182f56a8c9` | `ba57e8a386cd073e289814fc3f932b0f` |
| Treino (351 / 351) | `d7ee416586b457c1696481c80eeef4cc` | `ab6aef267000c247dae0b1ece4ccae85` |
| Teste (151 ham / 150 phishing) | `c2aecb1bdebe9d1fc4b98740e7e08da6` | `ccfa3c3612628cdfcee16eab995a7d00` |

Com a máscara, mais um par de phishing passa a ser idêntico (eram diferentes só no endereço de e-mail). Isso muda a amostra de 501 phishing e, portanto, todos os conjuntos. Os hashes foram calculados com a mesma função `_impressao` do `rodar_lote.py`, num ambiente de teste com cópia dos módulos e dos dados do PC. O ambiente foi validado por reproduzir os hashes de 07/10 com a máscara desligada. **A confirmar** na primeira execução no PC. Hashes que não valem como referência: `188e30e3…` é da 1ª versão (`f0b438b0…`), gravada e substituída no mesmo dia, sem nenhum treino; `6f148fa9…` e `fc851739…` vêm de testes intermediários que nunca foram gravados.

#### 1.5.5 O que fica no texto (decisão do usuário)

Por decisão do usuário, o que não é link nem endereço de e-mail faz parte do conteúdo da mensagem e é mantido. Isso não é uma limitação da replicação: é o próprio processo descrito no artigo, que mascara só URLs e endereços de e-mail.

- **Domínios soltos** (ex.: "sf.net", "apple.com", "monkey.org"): podem ser outras referências. Explicam a maior parte dos 12,2% de "monkey" que restam nos phishing.
- **Nomes do destinatário** (ex.: "Dear jose"): "jose" resta em 11,2% dos phishing processados.
- **O placeholder em si** se distribui de forma um pouco desigual (47,1% dos phishing, 57,2% dos legítimos, que vêm de listas de discussão com muitos endereços). É consequência direta da máscara prevista no artigo.
- **Observação à parte**, para análise futura: parte das ocorrências de "monkey.org" vem de linhas `Received: … by l.monkey.org` coladas no **meio** do corpo de e-mails do Nazario (cabeçalhos encaminhados). O `remover_cabecalho_smtp` não as remove porque não estão no início do corpo.

#### 1.5.6 Consequências e plano

1. **Novo lote de referência:** repetir o lote 5 modelos × 6 execuções (≈ 14,5 h no PC, como o de 07/10) com a máscara ligada. Este passa a ser o protocolo fiel ao artigo.
2. **O lote de 07/10 não é descartado:** passa a ser a condição "sem máscara de e-mail". Os dois lotes juntos formam uma **ablação** que mede o efeito do artefato. Se a acurácia cair com a máscara, parte do desempenho de 07/10 vinha do atalho.
3. **Hashes no notebook:** o `philder.ipynb`/`philder_kaggle.ipynb` contém `HASHES_REFERENCIA` com os valores de 07/10. Com as máscaras, a checagem vai acusar diferença, e isso é **esperado**. Os valores novos são os da Seção 1.5.4. Atualizar esses valores, ou criar no `rodar_lote.py` uma opção para a condição sem máscara, são alterações de código separadas que dependem de consentimento.
4. As Seções 5 e 8 continuam descrevendo o lote de 07/10.

> **Situação em 02/11/2026:** itens 1 e 2 executados — lote de 01–02/11 concluído e adotado como resultado oficial (Seção 10); ablação com/sem máscara na Seção 10.4. O item 3 continua pendente.

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
2. **As 6 execuções não são amostras independentes.** O recall tem desvio-padrão zero: 5 das 6 execuções erram exatamente os mesmos 2 e-mails (1 FP + 1 FN), e a execução 2 erra 1 FP a mais. As curvas de perda diferem entre execuções (não-determinismo do treino), mas o modelo final converge para as mesmas decisões, porque dados, split e semente do `Trainer` (42) são fixos. O ± mede só o não-determinismo de execução, não a incerteza do modelo — os p-valores acima devem ser lidos com essa ressalva (pseudo-replicação). Reforça a pendência de variar `random_state` (split) e `seed` (treino). *Decisão posterior (início da Fase 3):* manter o split e as sementes fixos no protocolo principal e tratar a variação de sementes no estudo de ablação (Seção 11.5).
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

**09/10 (execuções complementares).** Três rodadas extras de MiniLM, ALBERT e MobileBERT confirmaram os grupos de 07/10: o MobileBERT repetiu exatamente a média (0,9856) e os 3 falsos negativos; ALBERT e MiniLM empataram (0,9790), mostrando que a ordem entre os dois não é estável. Um ALBERT com os hiperparâmetros do próprio artigo (lr 1e-5, batch 32, warmup 6%) treinou sem platô e sem picos de perda, mas sem ganho de acurácia numa rodada (0,9734). A preparação de uma execução no Kaggle ficou bloqueada pela verificação de telefone (Seção 9).

---

## 5. Comparação com o artigo — lote de 07/10 (referência até 01/11)

> **Registro histórico (02/11/2026).** Esta seção descreve o lote de 07/10, obtido **sem a máscara de endereços de e-mail** (Seção 1.5). O resultado oficial atual e sua comparação com o artigo estão nas Seções 10.2–10.5.

Referência até 01/11: lote final de 07/10/2026 — protocolo fiel ao artigo (10 épocas fixas, sem seleção no teste), undersampling 502/501 (1.003 amostras), split estratificado 70/30 (`random_state=42`), mesmos hiperparâmetros para todos os modelos (lr 5e-5, batch 8, `weight_decay=0.01`, `max_length=128`), 6 execuções por modelo, PC local (CPU Intel, 12 threads, sem GPU). Detalhes na Seção 8.

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
3. **Diferenças em relação ao artigo.** MiniLM (−2,5 e-mails) e MobileBERT (−1,4) ficaram abaixo do artigo, o que muda a ordem dos modelos intermediários (artigo: DistilBERT > TinyBERT > MobileBERT > MiniLM > ALBERT; replicação: TinyBERT ≈ DistilBERT > MobileBERT ≈ ALBERT > MiniLM). Entre esses três modelos as diferenças (1–1,5 e-mail) estão próximas do erro-padrão; a separação entre o grupo superior (≈ 2 erros) e o inferior (4–6 erros) é nítida. As causas prováveis — sobreajuste do MiniLM com 10 épocas fixas, erro sistemático do MobileBERT em 3 e-mails — estão na Seção 8.9. Os hiperparâmetros recomendados pelos autores de cada modelo (Seção 11.6) indicam que o protocolo único desfavorece em especial o ALBERT (learning rate 5× maior que o recomendado, sem warmup) e o MiniLM (10 épocas contra 3–5 recomendadas). *Atualização 09/10:* nas rodadas complementares, ALBERT e MiniLM empataram (0,9790 cada) e, somando os dois dias, a diferença entre eles cai para ≈ 0,7 e-mail — a ordem entre esses dois não é estável, e a comparação deve ser apresentada **por grupos** (Seção 9.2). Um ALBERT com os hiperparâmetros do artigo original treinou de forma estável, mas sem ganho de acurácia numa rodada (Seção 9.3).
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
- Estudo detalhado dos resultados: iniciado em 08/10. Pendências consolidadas na Seção 11.5.

---

## 9. Execuções complementares de 09/10/2026

### 9.1 Contexto

Depois do lote final de 07/10 (referência, Seção 8), foram feitas em 09/10 três frentes complementares. Nenhuma altera o resultado de referência:

1. **Rodadas extras do protocolo principal** para os três modelos do grupo inferior (MiniLM, ALBERT e MobileBERT), para verificar a estabilidade da ordem entre eles (Seção 9.2).
2. **Experimento de hiperparâmetros do ALBERT**, com a receita do artigo original do modelo (Seção 11.6), para testar se a instabilidade e o desempenho mais baixo do ALBERT vêm da configuração única do protocolo (Seção 9.3).
3. **Preparação da execução no Kaggle**, como alternativa de nuvem ao Colab (Seção 9.4).

### 9.2 Rodadas extras do protocolo principal (MiniLM, ALBERT, MobileBERT)

**Configuração:** idêntica ao lote final (Seção 8.5) — `rodar_lote.py`, protocolo "artigo", 502/501, PC local, mesmas versões. Resumos: `resultados/lote_09102026_074659.csv` (MiniLM, 3 rodadas, 07h47–09h10) e `resultados/lote_09102026_091907.csv` (ALBERT e MobileBERT, 3 rodadas cada, 09h19–13h22). A primeira tentativa do ALBERT (rodada 1) foi interrompida manualmente às 09h18, ainda no pré-processamento (`KeyboardInterrupt`), e não gerou resultado; por isso as rodadas válidas do ALBERT são 2, 3 e 4. Os três hashes de dados são os de referência em todas as 9 execuções; nenhum erro nos logs.

**Resultados por execução**

| Modelo | Exec. | Acc. | Prec. | Rec. | F1 | FP | FN | T. Treino bruto (s) | T. Aval. por época (s) | T. Treino líquido (s) | T. Teste (s) | Detecção (ms) | TES (bruto) | TES (líquido) | IES | RTDE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM | 1 | 0,9834 | 0,9739 | 0,9933 | 0,9835 | 4 | 1 | 1.656,12 | 99,70 | 1.556,41 | 9,24 | 30,71 | 0,000594 | 0,000632 | 0,1064 | 32,02 |
| MiniLM | 2 | 0,9767 | 0,9673 | 0,9867 | 0,9769 | 5 | 2 | 1.538,61 | 96,54 | 1.442,07 | 9,52 | 31,62 | 0,000635 | 0,000677 | 0,1026 | 30,89 |
| MiniLM | 3 | 0,9767 | 0,9613 | 0,9933 | 0,9770 | 6 | 1 | 1.624,28 | 98,04 | 1.526,25 | 9,86 | 32,77 | 0,000601 | 0,000640 | 0,0990 | 29,81 |
| ALBERT | 2 | 0,9834 | 0,9866 | 0,9800 | 0,9833 | 2 | 3 | 3.241,05 | 401,26 | 2.839,79 | 40,07 | 133,12 | 0,000303 | 0,000346 | 0,0245 | 7,39 |
| ALBERT | 3 | 0,9767 | 0,9864 | 0,9667 | 0,9764 | 2 | 5 | 3.248,28 | 405,49 | 2.842,79 | 42,36 | 140,73 | 0,000301 | 0,000344 | 0,0231 | 6,94 |
| ALBERT | 4 | 0,9767 | 0,9735 | 0,9800 | 0,9767 | 4 | 3 | 3.141,99 | 388,97 | 2.753,02 | 39,02 | 129,62 | 0,000311 | 0,000355 | 0,0250 | 7,54 |
| MobileBERT | 1 | 0,9867 | 0,9932 | 0,9800 | 0,9866 | 1 | 3 | 1.489,73 | 113,02 | 1.376,71 | 11,26 | 37,40 | 0,000662 | 0,000717 | 0,0877 | 26,38 |
| MobileBERT | 2 | 0,9867 | 0,9932 | 0,9800 | 0,9866 | 1 | 3 | 1.643,97 | 114,72 | 1.529,25 | 11,39 | 37,83 | 0,000600 | 0,000645 | 0,0866 | 26,08 |
| MobileBERT | 3 | 0,9834 | 0,9866 | 0,9800 | 0,9833 | 2 | 3 | 1.527,46 | 113,83 | 1.413,63 | 12,60 | 41,87 | 0,000644 | 0,000696 | 0,0780 | 23,49 |

**Médias ± desvio-padrão (n = 3 por modelo)**

| Modelo | Acurácia | Precisão | Recall | F1 | Erros (FP / FN) | T. Treino (s) | T. Teste (s) | Detecção (ms) | TES | IES | RTDE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileBERT | 0,9856 ± 0,0019 | 0,9910 ± 0,0038 | 0,9800 ± 0,0000 | 0,9855 ± 0,0019 | 4,3 (1,3 / 3,0) | 1.553,72 ± 80,40 | 11,75 ± 0,74 | 39,03 ± 2,46 | 0,000635 ± 0,000032 | 0,0841 ± 0,0053 | 25,32 ± 1,59 |
| ALBERT | 0,9790 ± 0,0038 | 0,9822 ± 0,0075 | 0,9756 ± 0,0077 | 0,9788 ± 0,0039 | 6,3 (2,7 / 3,7) | 3.210,44 ± 59,39 | 40,48 ± 1,71 | 134,49 ± 5,68 | 0,000305 ± 0,000005 | 0,0242 ± 0,0010 | 7,29 ± 0,31 |
| MiniLM | 0,9790 ± 0,0038 | 0,9675 ± 0,0063 | 0,9911 ± 0,0038 | 0,9791 ± 0,0038 | 6,3 (5,0 / 1,3) | 1.606,34 ± 60,77 | 9,54 ± 0,31 | 31,70 ± 1,03 | 0,000610 ± 0,000022 | 0,1027 ± 0,0037 | 30,91 ± 1,11 |

**Comparação com o lote final de 07/10 e agregação dos dois dias**

| Modelo | 07/10 (n = 6) | 09/10 (n = 3) | Dois dias (n = 9) | Erros médios em 301 (07/10 → 09/10) |
|---|---|---|---|---|
| MobileBERT | 0,9856 ± 0,0034 | 0,9856 ± 0,0019 | 0,9856 ± 0,0029 | 4,3 → 4,3 |
| ALBERT | 0,9839 ± 0,0044 | 0,9790 ± 0,0038 | 0,9823 ± 0,0047 | 4,8 → 6,3 |
| MiniLM | 0,9806 ± 0,0039 | 0,9790 ± 0,0038 | 0,9801 ± 0,0037 | 5,8 → 6,3 |

**Observações das curvas de perda (logs)**

- **ALBERT:** rodadas 2 e 3 sem platô; a rodada 2 teve perda de avaliação ≈ 0,21 nas épocas 1–4 antes de cair a ≈ 0,10; a rodada 4 ficou no platô em ln 2 **apenas na época 1** (perda de avaliação 0,679, acurácia 0,498) e saiu na época 2. Perda de avaliação final mais alta que em 07/10 (até ≈ 0,20). Contagem de platôs do ALBERT no PC local: 1 de 1 no piloto (02/10), 0 de 6 em 07/10, 1 transitório de 3 em 09/10.
- **MobileBERT:** picos de perda de avaliação no início (1,50 na época 2 da rodada 2; 11,03 na época 1 da rodada 3), sempre recuperados — mesma instabilidade inicial já registrada (Seção 6.4).
- **MiniLM:** mesmo padrão de sobreajuste — na rodada 2, acurácia de avaliação 0,9867 nas épocas 7–8 e queda para 0,9767 na época 10, com perda de avaliação subindo de 0,086 para 0,145.

**Leitura**

1. **MobileBERT é o modelo mais reprodutível do grupo:** mesma média nos dois dias e os **mesmos 3 falsos negativos nas 9 execuções** — forte indício de que são sempre os mesmos 3 e-mails de phishing (prioridade da análise de erros).
2. **ALBERT × MiniLM: ordem instável.** Em 07/10 o ALBERT ficou à frente (0,9839 contra 0,9806); em 09/10 empataram exatamente (5, 7 e 7 erros em cada). Somando os dois dias, a ordem de 07/10 se mantém (0,9823 contra 0,9801), mas a diferença é de ≈ 0,7 e-mail, bem abaixo do erro-padrão binomial (≈ 1,5 e-mail). A comparação deve ser apresentada **por grupos**: superior (TinyBERT, DistilBERT, ≈ 2 erros), intermediário (MobileBERT, ≈ 4) e inferior (ALBERT e MiniLM, 5–6), e não como um ranking estrito.
3. **Perfis de erro distintos:** o MiniLM erra sobretudo por **falso positivo** (5,0 FP contra 1,3 FN), o MobileBERT por **falso negativo** (3,0 FN constantes), o ALBERT de forma mista.
4. **Tempos:** iguais aos de 07/10 dentro da variação da máquina (ALBERT 3.210 s contra 3.170 s; MobileBERT 1.554 s contra 1.581 s; MiniLM 1.606 s contra 1.574 s); TES, IES e RTDE também.
5. **Status:** estas execuções **complementam** o lote de 07/10 e não substituem a referência; o TinyBERT e o DistilBERT não foram repetidos em 09/10.

### 9.3 Experimento complementar: ALBERT com os hiperparâmetros do artigo original

**Motivação.** O levantamento da Seção 11.6 mostrou que o artigo do ALBERT usa, na tarefa mais parecida com a nossa (SST-2, Tabela 14), learning rate 1e-5, batch 32 e warmup de ≈ 6% — contra 5e-5, batch 8 e nenhum warmup no protocolo principal. O teste verifica se essa diferença explica a instabilidade (platô em ln 2, picos de perda) e o desempenho mais baixo do ALBERT.

**Desenho.** Configuração **fixada antes da execução**, a partir do artigo, sem nenhuma escolha feita olhando o teste (não há viés de seleção). Só três parâmetros mudam; o resto é idêntico ao protocolo principal.

| | Protocolo principal | Experimento |
|---|---|---|
| Learning rate | 5e-5 | **1e-5** |
| Batch de treino | 8 | **32** |
| Warmup | nenhum | **13 de 220 passos (6%)** |
| Épocas / `weight_decay` / `max_length` / batch de avaliação | 10 / 0,01 / 128 / 8 | iguais |
| Dados / split / semente | 502/501, estratificado, 42 | iguais (hash do teste `c2aecb1b…` conferido) |

**Implementação.** Célula independente acrescentada ao final do `philder.ipynb` ("EXPERIMENTO COMPLEMENTAR"), que monta `TrainingArguments` próprios e reutiliza as funções do pacote `philder` sem alterá-las. Saídas separadas do protocolo principal: `resultados/experimento_hp/albert/resultado_final-1_09102026.csv` e `curva_perda-1_09102026.png` (o CSV registra os hiperparâmetros e os hashes). Execução no PC local, 20h48–21h38 de 09/10, uma rodada.

**Evolução por época**

| Época | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Perda de treino | 0,293 | 0,182 | 0,048 | 0,025 | 0,004 | 0,003 | 0,002 | 0,002 | 0,002 | 0,002 |
| Perda de avaliação | 0,302 | 0,168 | 0,078 | **0,071** | 0,089 | 0,075 | 0,088 | 0,096 | 0,097 | 0,095 |
| Acurácia | 0,9468 | 0,9435 | 0,9767 | **0,9834** | 0,9734 | 0,9801 | 0,9734 | 0,9734 | 0,9734 | 0,9734 |
| F1 | 0,9456 | 0,9453 | 0,9767 | 0,9834 | 0,9737 | 0,9801 | 0,9737 | 0,9737 | 0,9737 | 0,9737 |

**Resultado final (modelo da 10ª época) e comparação**

| Métrica | ALBERT — hiperparâmetros do artigo (n = 1) | ALBERT — protocolo principal, 07/10 (n = 6) | ALBERT — protocolo principal, 09/10 (n = 3) |
|---|---|---|---|
| Acurácia | 0,9734 | 0,9839 ± 0,0044 | 0,9790 ± 0,0038 |
| Precisão | 0,9610 | 0,9771 ± 0,0079 | 0,9822 ± 0,0075 |
| Recall | 0,9867 | 0,9911 ± 0,0054 | 0,9756 ± 0,0077 |
| F1 | 0,9737 | 0,9840 ± 0,0044 | 0,9788 ± 0,0039 |
| Erros (FP / FN) | 8 (6 / 2) | 4,8 (3,5 / 1,3) | 6,3 (2,7 / 3,7) |
| T. Treino bruto (s) | 3.010,54 | 3.169,96 ± 40,33 | 3.210,44 ± 59,39 |
| T. Aval. por época (s) / T. Treino líquido (s) | 405,90 / 2.604,65 | 390,44 / 2.779,52 | — / 2.811,87 |
| T. Teste (s) | 40,46 | 39,11 ± 0,52 | 40,48 ± 1,71 |
| Detecção (ms) | 134,43 | 129,95 ± 1,72 | 134,49 ± 5,68 |
| TES (bruto) / TES (líquido) | 0,000323 / 0,000374 | 0,000310 / 0,000354 | 0,000305 / — |
| IES | 0,0241 | 0,0252 | 0,0242 |
| RTDE | 7,24 | 7,57 | 7,29 |
| Perda de avaliação final | 0,095 | ≈ 0,08–0,14 | até ≈ 0,20 |

**Leitura**

1. **A instabilidade desapareceu.** A curva de perda desce de forma suave, sem platô em ln 2 e sem os picos de 0,5–0,7 na perda de treino vistos em todas as rodadas do protocolo principal. O warmup com learning rate menor faz o que o artigo do ALBERT sugere.
2. **A acurácia não melhorou** nesta rodada: 8 erros, 1 a 3 a mais que as médias do protocolo principal — diferença dentro do erro-padrão (≈ 1,5 e-mail). Com uma rodada, não se pode afirmar piora; afirma-se apenas **ausência de ganho visível**.
3. **Sobreajuste após a época 4:** a perda de avaliação mínima (0,071) e a acurácia máxima (0,9834) ocorreram na época 4; depois o modelo estabiliza em 0,9734. A receita do artigo foi calibrada para o SST-2 (≈ 21 mil passos); aqui são 220 passos num conjunto pequeno. **Escolher 4 épocas a partir deste resultado seria viés de seleção no teste** (Seção 3.4); só uma validação separada do treino justificaria essa escolha.
4. **Treino ≈ 6% mais rápido** (3.011 s contra ≈ 3.200 s), provavelmente pelo melhor aproveitamento da CPU com batch 32; o tempo de teste não muda (batch de avaliação mantido em 8).
5. **Para o trabalho:** os indícios apontam que os hiperparâmetros explicam a **instabilidade** do ALBERT, mas não a **distância de acurácia** para o TinyBERT — o que reforça a conclusão central. É um indício de **uma rodada**; para virar resultado, faltam ao menos mais duas rodadas com a mesma configuração (Seção 11.5).

### 9.4 Preparação da execução no Kaggle

**Motivação.** Oferecer uma alternativa de nuvem ao Colab para rodar lotes sem depender do PC local.

**O que foi feito (fora do repositório; nenhum arquivo do projeto alterado):**

- A célula de montagem do Google Drive usada no Colab (`google.colab.drive.mount`) falha no Kaggle com `NotImplementedError` — o recurso só existe no Colab.
- Foi criado um notebook próprio, `philder_kaggle.ipynb`, com três formas de obter o código: (a) **GitHub** — `git clone` do repositório `thuatus/ml-paper-replication-phishing-detect-system` fixado em commit (padrão `da375111`, push de 08/10), com token lido dos *Secrets* do Kaggle só no cabeçalho da conexão (não fica gravado nem é impresso); (b) zip do projeto no **Google Drive** via `gdown`; (c) zip como **Kaggle Dataset**. Os dados brutos (excluídos do repositório pelo `.gitignore`) vão num Dataset à parte, `dados_philder.zip`.
- Antes de treinar, o notebook confere o MD5 dos dados brutos (`Nazario.csv`: `4022d055bb7cf8602f30f768f652e91e`; `20030228_easy_ham.tar.bz2`: `b323cb71981a88567266ebd76e025860`), os três hashes dos dados processados (referência 502/501 ou 501/501) e se a cópia do repositório está sem alterações locais; ao final grava `proveniencia_kaggle_<data>.json` e `resumo_kaggle_<data>.csv` com o commit usado.
- **Validação:** o caminho completo (clone, checkout, proveniência, MD5 e hashes 502/501) foi simulado com sucesso fora do Kaggle, com uma cópia local do repositório; o preparo de dados em Linux reproduziu os hashes de referência 502/501 e 501/501. Não foi possível testar a autenticação real no repositório privado nem as células de treino.
- Correção após o primeiro uso: os comandos `git` passaram a falhar imediatamente (e com limite de 5 minutos) quando não há credencial válida, em vez de ficarem esperando usuário e senha.

**Situação.** Execução **bloqueada**: no Kaggle, o acesso à internet (necessário para baixar os modelos do Hugging Face e clonar o GitHub) exige verificação de telefone, e o código SMS não chegou. Alternativas: Colab (já validado, Seção 7) ou Kaggle sem internet, com os modelos e o projeto enviados como Datasets (Seção 11.5).

## 10. Lote oficial com máscaras de links e e-mails (01–02/11/2026)

> **Resultado de referência oficial da replicação a partir de 02/11/2026**, por decisão do usuário. Substitui o lote de 07/10 (Seção 8), que passa a ser a condição "sem máscara" do estudo de ablação (Seção 10.4).

### 10.1 Configuração e execução

- **Código:** `philder/data_processing.py` com `mascarar_links_e_emails=True` (padrão; md5 `4c8eabe5…`, Seção 1.5.3). Os módulos A e C, o `rodar_lote.py` e os hiperparâmetros são os mesmos do lote de 07/10.
- **Validação prévia (01/11, 19h42):** uma rodada do TinyBERT (`python rodar_lote.py --modelos tinybert --rodadas 1`) confirmou a deduplicação (phishing 1.565 → 1.471) e os hashes previstos. Acurácia 0,9934 (2 FP / 0 FN); treino de 467,7 s, inflado por uso do PC durante a execução. Por decisão do usuário, os arquivos foram renomeados com o prefixo `valida_` (`resultados/tinybert/valida_*_01112026.*` e `resultados/valida_lote_01112026_194231.csv`) e **não entram** nas médias.
- **Comando:** `python rodar_lote.py --modelos todos --rodadas 6` (undersampling padrão `artigo`, 502/501). Resumo: `resultados/lote_01112026_195848.csv`.
- **Execução:** 30 treinamentos, **30 com status "ok"**, um processo por treinamento, das 19h58 de 01/11 às 09h47 de 02/11 (13,8 h de processamento). Ordem: TinyBERT, MiniLM, DistilBERT, MobileBERT, ALBERT. Acompanhamento de hora em hora pelos logs, sem nenhum erro ou *traceback*.
- **Ambiente (igual nas 30 execuções e ao lote de 07/10):** PC local, Windows, "Intel64 Family 6 Model 158" (12 threads), sem GPU; Python 3.12.13, torch 2.14.0+cpu, transformers 5.16.1.
- **Protocolo:** fiel ao artigo (10 épocas fixas, sem early stopping e sem seleção de checkpoint); lr 5e-5, batch 8, `weight_decay=0.01`, `max_length=128` para todos os modelos; split estratificado 70/30, `random_state=42`.
- **Dados (impressão digital MD5, idêntica nas 30 execuções):**

| Conjunto | Tamanho | Hash |
|---|---|---|
| Balanceado | 1.003 (502 ham / 501 phishing) | `ba57e8a386cd073e289814fc3f932b0f` |
| Treino | 702 (351 / 351) | `ab6aef267000c247dae0b1ece4ccae85` |
| Teste | 301 (151 ham / 150 phishing) | `ccfa3c3612628cdfcee16eab995a7d00` |

### 10.2 Tabela consolidada (média ± desvio-padrão, n = 6 por modelo)

**Desempenho preditivo**

| Modelo | Acurácia | Precisão | Recall | F1 | Erros em 301 (FP / FN) |
|---|---|---|---|---|---|
| **TinyBERT (PHILDER)** | **0,9939 ± 0,0039** | 0,9902 ± 0,0089 | 0,9978 ± 0,0034 | 0,9939 ± 0,0038 | 1,8 (1,5 / 0,3) |
| DistilBERT | 0,9906 ± 0,0044 | 0,9837 ± 0,0106 | 0,9978 ± 0,0034 | 0,9906 ± 0,0043 | 2,8 (2,5 / 0,3) |
| MobileBERT | 0,9906 ± 0,0033 | 0,9879 ± 0,0064 | 0,9933 ± 0,0000 | 0,9906 ± 0,0032 | 2,8 (1,8 / 1,0) |
| MiniLM | 0,9884 ± 0,0028 | 0,9814 ± 0,0063 | 0,9956 ± 0,0034 | 0,9884 ± 0,0027 | 3,5 (2,8 / 0,7) |
| ALBERT | 0,9878 ± 0,0054 | 0,9815 ± 0,0111 | 0,9944 ± 0,0027 | 0,9879 ± 0,0053 | 3,7 (2,8 / 0,8) |

**Eficiência computacional (mesmo PC, CPU)**

| Modelo | T. treino bruto (s) | T. treino líquido (s) | T. teste (s) | Detecção (ms) | TES (bruto) | TES (líquido) | IES | RTDE | Treino × TinyBERT | Teste × TinyBERT |
|---|---|---|---|---|---|---|---|---|---|---|
| **TinyBERT (PHILDER)** | 392,3 ± 4,4 | 370,1 ± 3,9 | 2,20 ± 0,02 | 7,31 ± 0,08 | 0,00253 ± 0,00002 | 0,00269 ± 0,00002 | 0,452 ± 0,005 | 136,0 ± 1,5 | 1,00× | 1,00× |
| DistilBERT | 1.699,6 ± 7,8 | 1.539,9 ± 7,9 | 16,25 ± 0,36 | 53,99 ± 1,19 | 0,00058 ± 0,00000 | 0,00064 ± 0,00001 | 0,061 ± 0,001 | 18,4 ± 0,4 | 4,33× | 7,39× |
| MobileBERT | 1.466,6 ± 50,7 | 1.361,4 ± 50,5 | 10,86 ± 0,76 | 36,09 ± 2,52 | 0,00068 ± 0,00002 | 0,00073 ± 0,00003 | 0,092 ± 0,006 | 27,5 ± 1,7 | 3,74× | 4,94× |
| MiniLM | 1.430,8 ± 12,7 | 1.339,1 ± 12,4 | 9,10 ± 0,07 | 30,23 ± 0,23 | 0,00069 ± 0,00001 | 0,00074 ± 0,00001 | 0,109 ± 0,001 | 32,7 ± 0,3 | 3,65× | 4,14× |
| ALBERT | 3.085,0 ± 11,1 | 2.708,8 ± 10,3 | 37,40 ± 0,19 | 124,25 ± 0,64 | 0,00032 ± 0,00000 | 0,00036 ± 0,00000 | 0,026 ± 0,000 | 8,0 ± 0,1 | 7,86× | 17,00× |

### 10.3 Resultados por execução

| Modelo | Rodada | Acurácia | Precisão | Recall | F1 | FP | FN | T. treino (s) | T. teste (s) | Detecção (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| TinyBERT (PHILDER) | 1 | 0,9967 | 0,9934 | 1,0000 | 0,9967 | 1 | 0 | 398,3 | 2,19 | 7,27 |
| TinyBERT (PHILDER) | 2 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 388,9 | 2,20 | 7,30 |
| TinyBERT (PHILDER) | 3 | 0,9967 | 0,9934 | 1,0000 | 0,9967 | 1 | 0 | 397,4 | 2,18 | 7,25 |
| TinyBERT (PHILDER) | 4 | 0,9934 | 0,9868 | 1,0000 | 0,9934 | 2 | 0 | 391,4 | 2,24 | 7,45 |
| TinyBERT (PHILDER) | 5 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 0 | 1 | 388,7 | 2,21 | 7,34 |
| TinyBERT (PHILDER) | 6 | 0,9867 | 0,9740 | 1,0000 | 0,9868 | 4 | 0 | 389,3 | 2,18 | 7,24 |
| DistilBERT | 1 | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 2 | 1 | 1.705,5 | 16,36 | 54,36 |
| DistilBERT | 2 | 0,9900 | 0,9804 | 1,0000 | 0,9901 | 3 | 0 | 1.694,9 | 16,91 | 56,18 |
| DistilBERT | 3 | 0,9834 | 0,9677 | 1,0000 | 0,9836 | 5 | 0 | 1.711,3 | 16,16 | 53,70 |
| DistilBERT | 4 | 0,9900 | 0,9804 | 1,0000 | 0,9901 | 3 | 0 | 1.700,3 | 15,98 | 53,10 |
| DistilBERT | 5 | 0,9934 | 0,9868 | 1,0000 | 0,9934 | 2 | 0 | 1.690,0 | 16,16 | 53,67 |
| DistilBERT | 6 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 0 | 1 | 1.695,4 | 15,93 | 52,92 |
| MobileBERT | 1 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 1.486,9 | 10,51 | 34,93 |
| MobileBERT | 2 | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 2 | 1 | 1.488,5 | 10,40 | 34,55 |
| MobileBERT | 3 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 1.493,5 | 12,35 | 41,02 |
| MobileBERT | 4 | 0,9867 | 0,9803 | 0,9933 | 0,9868 | 3 | 1 | 1.380,9 | 10,45 | 34,72 |
| MobileBERT | 5 | 0,9867 | 0,9803 | 0,9933 | 0,9868 | 3 | 1 | 1.518,1 | 10,49 | 34,84 |
| MobileBERT | 6 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 1.431,5 | 10,98 | 36,48 |
| MiniLM | 1 | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 2 | 1 | 1.447,2 | 8,99 | 29,85 |
| MiniLM | 2 | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 2 | 1 | 1.430,4 | 9,16 | 30,42 |
| MiniLM | 3 | 0,9834 | 0,9739 | 0,9933 | 0,9835 | 4 | 1 | 1.425,7 | 9,14 | 30,35 |
| MiniLM | 4 | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 2 | 1 | 1.434,8 | 9,14 | 30,38 |
| MiniLM | 5 | 0,9900 | 0,9804 | 1,0000 | 0,9901 | 3 | 0 | 1.409,4 | 9,03 | 30,02 |
| MiniLM | 6 | 0,9867 | 0,9740 | 1,0000 | 0,9868 | 4 | 0 | 1.437,1 | 9,13 | 30,33 |
| ALBERT | 1 | 0,9834 | 0,9739 | 0,9933 | 0,9835 | 4 | 1 | 3.095,3 | 37,47 | 124,48 |
| ALBERT | 2 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 3.074,8 | 37,68 | 125,17 |
| ALBERT | 3 | 0,9867 | 0,9740 | 1,0000 | 0,9868 | 4 | 0 | 3.079,7 | 37,29 | 123,88 |
| ALBERT | 4 | 0,9900 | 0,9868 | 0,9933 | 0,9900 | 2 | 1 | 3.085,2 | 37,12 | 123,31 |
| ALBERT | 5 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 1 | 1 | 3.073,8 | 37,34 | 124,06 |
| ALBERT | 6 | 0,9801 | 0,9675 | 0,9933 | 0,9803 | 5 | 1 | 3.101,0 | 37,50 | 124,57 |

Curvas de perda: `resultados/<modelo>/curva_perda-<rodada>_01112026.png`. Conferidas no acompanhamento: nenhuma execução teve o platô de acurácia ≈ 0,5 (o ALBERT treinou de forma estável nas 6). A perda de avaliação do MobileBERT continua em escala alta (≈ 5), com acurácia de 0,99 — mesmo comportamento de escala de 07/10 (Seção 6.5).

### 10.4 Ablação: com máscaras (01/11) vs. sem máscara (07/10)

| Modelo | Acurácia sem máscara | Acurácia com máscaras | Δ (p.p.) | Δ (e-mails em 301) | Recall sem → com | Precisão sem → com | FP sem → com | FN sem → com | T. treino sem → com (s) |
|---|---|---|---|---|---|---|---|---|---|
| TinyBERT (PHILDER) | 0,9934 ± 0,0021 | 0,9939 ± 0,0039 | +0,06 | +0,2 | 0,9933 → 0,9978 | 0,9934 → 0,9902 | 1,0 → 1,5 | 1,0 → 0,3 | 407,4 → 392,3 |
| DistilBERT | 0,9928 ± 0,0014 | 0,9906 ± 0,0044 | −0,22 | −0,7 | 0,9933 → 0,9978 | 0,9923 → 0,9837 | 1,2 → 2,5 | 1,0 → 0,3 | 1.806,9 → 1.699,6 |
| MobileBERT | 0,9856 ± 0,0034 | 0,9906 ± 0,0033 | +0,50 | +1,5 | 0,9800 → 0,9933 | 0,9911 → 0,9879 | 1,3 → 1,8 | 3,0 → 1,0 | 1.581,4 → 1.466,6 |
| MiniLM | 0,9806 ± 0,0039 | 0,9884 ± 0,0028 | +0,78 | +2,3 | 0,9822 → 0,9956 | 0,9790 → 0,9814 | 3,2 → 2,8 | 2,7 → 0,7 | 1.574,3 → 1.430,8 |
| ALBERT | 0,9839 ± 0,0044 | 0,9878 ± 0,0054 | +0,39 | +1,2 | 0,9911 → 0,9944 | 0,9771 → 0,9815 | 3,5 → 2,8 | 1,3 → 0,8 | 3.170,0 → 3.085,0 |

Somando os cinco modelos (média por rodada): FN 9,0 → 3,2; FP 10,2 → 11,5.

**Cuidado na leitura:** a máscara mudou a deduplicação (um par de phishing a mais ficou idêntico) e, com isso, a amostra de 501 phishing e os conjuntos de treino e teste. Os dois lotes **não** usam o mesmo teste, então a comparação vale para as médias, não e-mail a e-mail. Com 301 e-mails, o erro-padrão binomial é de cerca de ±0,5 p.p.

### 10.5 Comparação com o artigo

**PHILDER (TinyBERT) vs. artigo**

| Métrica | Artigo (TinyBERT (2), Tab. 3 / Tab. 4; eficiência pela fórmula bruta) | Replicação oficial (01/11, n = 6) | Diferença |
|---|---|---|---|
| Acurácia | 0,9927 ± 0,0006 | 0,9939 ± 0,0039 | +0,12 p.p. (≈ +0,4 e-mail) |
| Precisão | 0,9896 | 0,9902 ± 0,0089 | +0,06 p.p. |
| Recall | 0,9960 | 0,9978 ± 0,0034 | +0,18 p.p. |
| F1 | 0,9927 ± 0,0006 | 0,9939 ± 0,0038 | +0,12 p.p. |
| Erros em 301 (FP / FN) | — | 1,8 (1,5 / 0,3) | — |
| T. treino bruto (s) | 180,52 | 392,31 ± 4,38 | 2,17× mais lento |
| T. teste (s) | 6,22 | 2,20 ± 0,02 | 2,83× mais rápido |
| Detecção (ms/amostra) | 12,9811 | 7,31 ± 0,08 | 1,78× mais rápido |
| TES (bruto) | 0,0055 | 0,002534 ± 0,000025 | 2,17× menor |
| IES | 0,1596 | 0,4519 ± 0,0050 | 2,83× maior |
| RTDE | 76,47 | 136,02 ± 1,50 | 1,78× maior |

**Os cinco modelos vs. artigo**

| Modelo | Acc. artigo | Acc. replicação oficial | Δ (e-mails em 301) | Rec. artigo | Rec. replicação | T. treino artigo → replicação (s) | T. teste artigo → replicação (s) |
|---|---|---|---|---|---|---|---|
| TinyBERT (PHILDER) | 0,9927 ± 0,0006 | 0,9939 ± 0,0039 | +0,4 | 0,9960 | 0,9978 | 180,52 → 392,3 | 6,22 → 2,20 |
| DistilBERT | 0,9934 ± 0,0005 | 0,9906 ± 0,0044 | −0,8 | 1,0000 | 0,9978 | 111,83 → 1.699,6 | 4,48 → 16,25 |
| MobileBERT | 0,9904 ± 0,0004 | 0,9906 ± 0,0033 | +0,1 | 0,9865 | 0,9933 | 219,45 → 1.466,6 | 9,76 → 10,86 |
| MiniLM | 0,9889 ± 0,0007 | 0,9884 ± 0,0028 | −0,2 | 0,9982 | 0,9956 | 103,52 → 1.430,8 | 6,21 → 9,10 |
| ALBERT | 0,9834 ± 0,0008 | 0,9878 ± 0,0054 | +1,3 | 0,9864 | 0,9944 | 236,81 → 3.085,0 | 9,09 → 37,40 |

### 10.6 Achados

1. **O PHILDER continua reproduzido, agora com o pré-processamento completo do artigo.** O TinyBERT tem acurácia 0,9939 ± 0,0039, a +0,12 p.p. do artigo (≈ 0,4 e-mail em 301), dentro do erro-padrão binomial (≈ ±0,5 p.p.). É a maior acurácia entre os cinco modelos e não dependia do atalho `jose@monkey.org`: sem a máscara, o resultado era praticamente o mesmo (Seção 10.4).
2. **A replicação ficou mais próxima do artigo nos cinco modelos.** Em 07/10, MiniLM (−2,5 e-mails) e MobileBERT (−1,4) ficavam abaixo do artigo (Seção 5.2). Com as máscaras, os cinco modelos ficam a no máximo 1,3 e-mail das médias do artigo: TinyBERT +0,4, MobileBERT +0,1, MiniLM −0,2, DistilBERT −0,8, ALBERT +1,3. Parte das diferenças de 07/10 vinha, portanto, do pré-processamento incompleto, e não só dos hiperparâmetros.
3. **A afirmação central do artigo se sustenta.** O TinyBERT é o mais preciso e o mais barato em todas as medidas de tempo: treino 3,7–7,9× mais rápido e teste 4,1–17,0× mais rápido que os demais, no mesmo hardware (Seção 10.2).
4. **As máscaras mudaram o tipo de erro.** Nos cinco modelos, o recall subiu e os falsos negativos caíram (soma por rodada: 9,0 → 3,2), enquanto a precisão caiu um pouco em três modelos e os falsos positivos subiram levemente (10,2 → 11,5). Na detecção de phishing, deixar escapar menos ataques costuma ser o erro mais importante de evitar. Hipótese a verificar na análise de erros: com `max_length=128`, um endereço que antes ocupava vários tokens ("jose monkey org", nomes de listas) agora ocupa um só, sobrando mais espaço para o conteúdo; e o `EMAILADDRESS`, mais frequente nos legítimos de listas de discussão (Seção 1.5.5), pode pesar nos FP.
5. **MobileBERT: de 3 FN constantes para 1 FN constante.** Em 07/10, o MobileBERT perdia sempre os mesmos 3 phishing; agora perde exatamente 1 nas 6 rodadas (recall 0,9933 ± 0,0000). É o caso mais claro de efeito da máscara e o primeiro candidato da análise de erros (quais e-mails, e se continham endereços ou links).
6. **Ordem dos modelos (por grupos).** TinyBERT (1,8 erro por rodada) > DistilBERT ≈ MobileBERT (2,8) > MiniLM ≈ ALBERT (3,5–3,7). Os quatro últimos estão a menos de 1 e-mail entre si; como na Seção 9.2, a comparação deve ser apresentada por grupos, não como ranking estrito.
7. **Estabilidade.** Nenhum platô em 30 execuções. Os desvios-padrão da acurácia (0,0028–0,0054) são um pouco maiores que em 07/10 em três modelos, puxados por rodadas isoladas com 4–5 FP (TinyBERT rodada 6, DistilBERT rodada 3, ALBERT rodada 6).
8. **Eficiência.** Os tempos de treino ficaram 3–9% menores que em 07/10 em todos os modelos, com a mesma máquina e as mesmas versões. A causa mais provável é carga do PC (o lote rodou à noite, sem uso), e não a máscara. O padrão em relação ao artigo se mantém: treino ≈ 2,2× mais lento e teste ≈ 2,8× mais rápido no TinyBERT (Seção 5.3, item 4).
9. **Ressalva estatística.** Como em 07/10, as 6 execuções variam só o não-determinismo do treino, com dados e teste fixos: os desvios medem a estabilidade do fine-tuning, não a incerteza sobre dados novos (Seção 5.3, item 5).

### 10.7 Status

- **Resultado de referência oficial a partir de 02/11/2026:** este lote (Seções 10.2–10.5). O lote de 07/10 (Seções 5 e 8) passa a ser registro histórico e condição "sem máscara" da ablação.
- **Pendente (depende de consentimento):** atualizar `HASHES_REFERENCIA` nos notebooks `philder.ipynb` e `philder_kaggle.ipynb` para os hashes desta seção.
- **Próximos passos:** análise de erros (FN do MobileBERT, FP de todos os modelos), checagem de quase-duplicatas treino/teste e demais itens da Seção 11.5.

---

## 11. Conclusão e próximos passos

### 11.1 Síntese da trajetória

**Fase 2 (17/09–25/09) — corrigir os dados.** A replicação começou com resultados "bons demais" (precisão 1,0000), que levaram ao diagnóstico de vazamento de dados (duplicatas entre treino e teste). A correção passou por uma solução desenhada mas nunca executada (23/09) e pela descoberta de dois bugs reais de implementação (código morto no Módulo A e rótulos invertidos, 24/09). O lote 25/09(b) foi o primeiro com pipeline verificado linha a linha e sobreposição treino/teste comprovadamente nula (Seções 1.2, 3 e 4).

**Fase 3 (01/10–07/10) — corrigir o método e ampliar a comparação.** (1) Remoção do viés de seleção de modelo no teste, com protocolo fiel ao artigo (01/10, Seção 3.4); (2) correção da tabela de referência do artigo (Seção 2); (3) expansão para os cinco modelos do artigo (02/10, Seção 6); (4) reprodução em outro ambiente e correção da dependência da ordem de leitura dos arquivos (Colab, 03–04/10, Seção 7); (5) automação da execução com registro de ambiente e impressões digitais (04/10, Seção 8.2); (6) alinhamento do undersampling ao artigo, 502/501 (07/10, Seção 8.4); (7) lote final com 5 modelos × 6 execuções (07/10, Seções 8.5–8.10).

**Fase 4 (01/11–02/11) — completar o pré-processamento do artigo.** A inspeção da saída real do Módulo B mostrou que o endereço da caixa-isca do Nazario (`jose@monkey.org`) virava palavras soltas em ≈ 40% dos phishing — um atalho para o modelo —, porque os endereços de e-mail não eram mascarados, ao contrário do artigo (Seção 1.5). As máscaras de links e e-mails foram completadas e verificadas nos dois corpora, e o lote de 5 modelos × 6 execuções foi refeito (Seção 10), tornando-se o resultado oficial.

### 11.2 Resultado de referência atual

**Lote oficial de 01–02/11/2026 (Seção 10), adotado em 02/11/2026.** Com o pré-processamento completo do artigo (máscaras de links e e-mails), o TinyBERT (PHILDER) tem Acurácia 0,9939 ± 0,0039 e F1 0,9939 ± 0,0038, contra 0,9927 ± 0,0006 no artigo (+0,12 p.p., ≈ 0,4 e-mail em 301) — o desempenho preditivo do PHILDER está reproduzido. É o modelo mais preciso dos cinco e o mais eficiente em todas as medidas: treino 3,7–7,9× mais rápido e teste 4,1–17,0× mais rápido que os demais, no mesmo hardware. A afirmação central do artigo — o TinyBERT oferece o melhor equilíbrio entre acurácia e custo — se sustenta. Os cinco modelos ficam a no máximo 1,3 e-mail das médias do artigo (Seção 10.5), mais perto do que em 07/10. Em grupos: TinyBERT > DistilBERT ≈ MobileBERT > MiniLM ≈ ALBERT (Seção 10.6). A ablação com o lote de 07/10 mostra que o TinyBERT não dependia do atalho `jose@monkey.org` e que as máscaras reduzem os falsos negativos em todos os modelos (Seção 10.4).

**Registro histórico — referência de 07/10 a 01/11.** Lote final de 07/10 (Seção 8), sem a máscara de endereços de e-mail: TinyBERT (PHILDER) com Acurácia 0,9934 ± 0,0021 e F1 0,9933 ± 0,0021, contra 0,9927 ± 0,0006 no artigo — o desempenho preditivo do PHILDER foi reproduzido. O TinyBERT empata com o DistilBERT (0,9928 ± 0,0014), o modelo mais preciso, e é o mais eficiente em todas as medidas: 3,9–7,8× mais rápido no treino e 4,2–17,1× na inferência que os demais modelos, no mesmo hardware. A afirmação central do artigo — o TinyBERT oferece o melhor equilíbrio entre acurácia e custo — se sustenta, e com margem maior que a sugerida pela Tabela 3 do próprio artigo. As diferenças em relação ao artigo (MiniLM e MobileBERT abaixo; tempos de treino maiores e de inferência menores) estão registradas na Seção 5.3. As execuções complementares de 09/10 (Seção 9) confirmam a separação em três grupos e mostram que a ordem entre ALBERT e MiniLM não é estável; o ALBERT com os hiperparâmetros do próprio artigo treinou de forma estável, mas sem ganho de acurácia numa rodada.

### 11.3 Lições metodológicas

1. **Plausibilidade das métricas não substitui verificação direta** do código executado e dos dados resultantes (vazamento de dados e código morto, Seção 1.2).
2. **Usar o teste para decidir o treino infla o resultado,** mesmo sem nenhum e-mail do teste nos pesos (viés de seleção, ≈ 0,4 p.p.; Seção 3.4).
3. **Contagens iguais não provam conteúdo igual;** sementes fixas não garantem reprodutibilidade se a ordem de entrada dos dados depender do sistema operacional (impressão digital MD5, Seção 7.2).
4. **Uma gravação "concluída sem erro" não prova que o conteúdo certo chegou ao destino** — só a leitura do arquivo resultante prova (Seção 6.2).
5. **Repetir execuções com os mesmos dados mede a estabilidade do treino, não a incerteza do resultado** (pseudo-replicação, Seções 3.4 e 8.9); com n = 301, diferenças de 1–2 e-mails não são distinguíveis (erro-padrão ≈ ±0,5 p.p.).
6. **Métricas de eficiência só são comparáveis no mesmo hardware,** que deve ser reportado junto com elas; a vantagem do modelo menor cresce em hardware mais fraco (Seções 7.5 e 7.9).
7. **Instabilidade de fine-tuning precisa de várias execuções para aparecer** (platô do ALBERT: 1 de 1 no piloto, 0 de 6 no lote final; Seções 7.8 e 8.9).
8. **Duplicatas podem ser invisíveis no texto bruto:** todas as 80 duplicatas descartadas nos dois corpora só aparecem depois do pré-processamento (Seção 1.4); a comparação deve usar o mesmo texto que o modelo recebe.
9. **Conferir cada etapa do pré-processamento contra a descrição do artigo, na saída real:** a falta da máscara de endereços de e-mail só apareceu ao ler um `body_processado` (Seção 1.5). Endereços, domínios e nomes ligados à fonte de coleta criam atalhos que o teste da mesma distribuição não revela.
10. **Corrigir um atalho e manter o resultado é evidência, não decepção:** com as máscaras, a acurácia do TinyBERT ficou igual e a dos outros modelos subiu ou caiu menos de 1 e-mail (Seção 10.4). Isso mostra que o desempenho vinha do conteúdo das mensagens, não do endereço da caixa-isca. A comparação só é possível porque o lote antigo foi mantido como condição de ablação.

### 11.4 Registro histórico — conclusão de 25/09

Conclusão registrada ao fim da Fase 2, mantida para registro. O resultado de referência nela citado (25/09(b)) foi substituído (Seção 1.3).

O resultado de 25/09(b) é adotado como referência definitiva da replicação: metodologicamente correto (sem vazamento de dados, confirmado via checagem programática explícita de sobreposição treino/teste, não apenas pela plausibilidade das métricas), com ambos os pools do dataset (legítimos/SpamAssassin e phishing/Nazario) deduplicados antes da amostragem, e com o pipeline de execução do notebook verificado linha a código — o primeiro da replicação a atingir esse nível de certeza. O lote 25/09(a), mantido lado a lado na Seção 3, documenta o passo intermediário (dedup só do phishing) e permite isolar o efeito específico de estender a deduplicação também ao lado dos legítimos (Seção 4). A evolução registrada nesta linha do tempo — da detecção inicial de um resultado "bom demais" (21/09), passando por uma correção desenhada mas nunca de fato executada (23/09), até a descoberta dos dois bugs reais de implementação e a comparação entre as duas técnicas de deduplicação validadas em 25/09(a) e (b) — é, em si, material especialmente relevante para a entrega de "análise crítica dos métodos de aferição de desempenho e eficácia" prevista na Fase 2: a lição central é que plausibilidade de métricas não substitui verificação direta do código executado e dos dados resultantes.

Itens em aberto para próximas atualizações deste documento:
- Repetir a execução de referência (25/09(b)) com sementes diferentes de `random_state` (todas as 6 execuções atuais usam a mesma semente/configuração, variando apenas por não-determinismo de treino), para isolar variância de semente de variância de treino.
- Incorporar novos marcos à linha do tempo conforme forem executados.
- A extensão da deduplicação prévia ao lado do SpamAssassin (item que estava em aberto neste documento até a atualização anterior) já foi implementada e validada como parte da correção de 24–25/09 — item concluído, com os dois lotes (a) e (b) documentando o antes/depois dessa extensão.

> **Situação em 08/10:** o primeiro item (sementes) teve decisão — manter o split e as sementes fixos no protocolo principal e avaliar a variação de sementes no estudo de ablação (Seção 11.5). O segundo é atendido continuamente por este documento. O terceiro já estava concluído.

### 11.5 Pendências consolidadas (08/10/2026; atualizado em 02/11)

**Lote com máscaras de links e e-mails (01–02/11/2026; Seções 1.5 e 10)**
- ~~Executar o lote 5 modelos × 6 execuções com as máscaras e confirmar os novos hashes.~~ **Concluído em 02/11** (Seção 10; hashes confirmados nas 30 execuções).
- ~~Comparar com o lote de 07/10 e registrar o efeito do artefato `jose@monkey.org`.~~ **Concluído** (Seção 10.4).
- Análise de erros do lote oficial: o 1 phishing que o MobileBERT perde nas 6 execuções (antes eram 3), os FP de todos os modelos (subiram levemente com as máscaras) e o papel do placeholder `EMAILADDRESS` nesses FP (Seção 10.6, itens 4 e 5).
- Decidir (com consentimento) sobre atualizar `HASHES_REFERENCIA` nos notebooks e sobre uma opção no `rodar_lote.py` para a condição sem máscara.
- Opcional, para análise futura: linhas `Received:` coladas no meio do corpo de e-mails do Nazario (Seção 1.5.5). Domínios soltos e nomes ficam no texto, por decisão do usuário.

**Estudo dos resultados**
- Estudo detalhado do lote final (iniciado em 08/10).
- Análise de erros: os 3 phishing que o MobileBERT perde em todas as execuções (FN constante — confirmado nas 9 execuções de 07/10 e 09/10); os FP do ALBERT e do MiniLM (o MiniLM erra sobretudo por FP, Seção 9.2); os e-mails errados por todas as execuções do TinyBERT em 01/10 (Seção 3.4, item 4).
- Conferir as curvas de perda de TinyBERT, MiniLM, DistilBERT e MobileBERT no lote final (só o ALBERT foi conferido nas 6 execuções; MiniLM em 1).
- Investigar o padrão "treino mais lento, inferência mais rápida" em relação ao artigo (comprimento de sequência, batch de inferência, tokenização dentro do tempo de teste; Seções 3.4 e 5.3).

**Experimentos complementares (separados do protocolo principal)**
- Estudo de ablação com variação de sementes (`random_state` do split e `seed` do treino).
- Warmup / learning rate menor para o ALBERT, sem usar o teste para escolher a configuração do protocolo principal (Seção 7.10). Ponto de partida documentado: Tabela 14 do artigo do ALBERT (Seção 11.6). *1ª rodada feita em 09/10 (Seção 9.3): treino estável, acurácia 0,9734.* Faltam ao menos mais 2 rodadas com a mesma configuração.
- Fine-tuning com os hiperparâmetros recomendados por modelo (faixas da Seção 11.6), escolhidos num conjunto de **validação separado do treino** e medidos no teste uma única vez — inclusive menos épocas para o MiniLM (3–5) e batch 16–32 para todos.
- Experimento GPU vs. CPU, para testar a hipótese de compressão das diferenças de tempo em hardware potente (Seções 6.4 e 7.9).
- Checagem de **quase-duplicatas** entre treino e teste (similaridade de texto, ex.: proporção de palavras em comum acima de 90%), motivada pelas campanhas repetidas do Nazario (Seção 1.4.4, item 5).
- Estudo de ablação do pré-processamento: treinar **com e sem remoção de stopwords** (a remoção apaga negações; Seção 1.4.4, item 3) e avaliar o tratamento de letras acentuadas.
- Kaggle: resolver a verificação de telefone (suporte do Kaggle) ou adaptar o `philder_kaggle.ipynb` para rodar **sem internet**, com os modelos (cache do Hugging Face do PC local) e o projeto enviados como Datasets (Seção 9.4).
- Comparação 501/501 vs. 502/501 em várias execuções, se for útil separar o efeito do dado do efeito do acaso.

**Código e reprodutibilidade (cada alteração de código depende de consentimento explícito)**
- Commit/push para o GitHub do estado validado (`rodar_lote.py`, `database_consolidation.py`, `database_balance.py`, `data_processing.py` — alterado em 01/11 —, notebook e documentos).
- `requirements.txt` com versões fixas (Seção 7.6).
- Curva de perda em escala logarítmica ou com recorte dos passos iniciais (MobileBERT ilegível em escala linear; Seção 6.5).
- O cabeçalho do `rodar_lote.py` cita "Analise_dos_Resultados.md, Seção 8.2", que após a reorganização de 08/10 passou a ser a Seção 7.2.

**Organização de arquivos e documentos**
- Mover `resultados/tinybert/resultado_final-7_07102026.csv` (cópia do lote de validação 501/501) para `resultados/tinybert/teste e validacao/`; decidir se o CSV da validação 502/501 (Seção 8.4) deve ser recuperado de `.ipynb_checkpoints`.
- Atualizar `Embasamento_Teorico_Metricas_PHILDER.md` (versão de 25/09) e o handoff do projeto com a referência atual.
- Remover cópias duplicadas deste documento no Projeto.
- Iniciar a redação do artigo (Overleaf, formato IEEE) a partir das Seções 5, 8, 9, 10 e 11, apresentando a comparação entre modelos **por grupos** (Seção 9.2).

### 11.6 Hiperparâmetros recomendados pelos autores dos modelos (levantamento de 09/10/2026)

**Motivação.** O protocolo deste trabalho usa os mesmos hiperparâmetros para os cinco modelos (lr 5e-5, batch 8, 10 épocas, `weight_decay=0.01`, `max_length=128`, sem warmup), por fidelidade ao PHILDER e para evitar viés de seleção no teste (Seção 3.4). Para avaliar se essa escolha prejudica algum modelo, foram consultados os model cards dos checkpoints no Hugging Face e os artigos originais de cada arquitetura.

**Model cards (Hugging Face).** Nenhum traz receita de fine-tuning:

| Checkpoint | Conteúdo do model card | Remete a |
|---|---|---|
| `albert/albert-base-v2` | Pré-processamento (SentencePiece, vocabulário de 30.000) e mascaramento; resultados sem as configurações usadas | Artigo [27] e repositório google-research/albert |
| `microsoft/MiniLM-L12-H384-uncased` | Aviso "needs to be fine-tuned before use"; resultados em SQuAD 2.0 e GLUE sem configurações | Repositório microsoft/unilm (minilm) |
| `distilbert` (base) | Pré-treino em 8 GPUs V100 de 16 GB por ≈ 90 h; tabela do GLUE sem configurações | Código de treino da Hugging Face |
| `google/mobilebert-uncased` | Nenhum hiperparâmetro; exemplo de uso | — |
| `huawei-noah/TinyBERT_General_4L_312D` | Card oficial não acessado; página que espelha o card não traz hiperparâmetros | Artigo [25] |

**Artigos originais.** As recomendações estão nos artigos e, em todos os casos, foram escolhidas num conjunto de **validação (dev)**:

| Fonte | Learning rate | Batch | Épocas / passos | Warmup | Comprimento máx. | Observação |
|---|---|---|---|---|---|---|
| BERT [22], Apêndice A.3 (referência geral) | 5e-5, 3e-5, 2e-5 | 16, 32 | 2, 3, 4 | — | — | dropout sempre 0,1 |
| ALBERT [27], Tabela 14 — SST-2 (classificação de frase única, a mais próxima deste trabalho) | **1e-5** | 32 | 20.935 passos | **1.256 passos (≈ 6%)** | 512 | dropout do ALBERT 0, do classificador 0,1; warmup em todas as tarefas da tabela |
| MiniLM [28], Apêndice B — GLUE | 2e-5 a 5e-5 | 32 | 3 a 5 (até 10 em CoLA, RTE e MRPC) | não informado | 128 | — |
| MobileBERT [26], Seção 4.2 | 1e-5 a 10e-5 | 16, 32, 48 | 2 a 10 | não informado | — | "usually needs a larger learning rate and more training epochs" |
| TinyBERT [25], Seção 4.2 | 1e-5 a 3e-5 (destilação da camada de predição) | 16, 32 | 20 (destilação intermediária) + 3 (predição) | — | 64 (frase única) / 128 (pares) | procedimento de **destilação com o BERT como professor**, não fine-tuning direto |
| DistilBERT [24] | não informa | — | — | — | — | só o pré-treino (lotes de até 4K exemplos, 8 V100, ≈ 90 h) |
| **Este trabalho** | 5e-5 | **8** | **10** | **nenhum** | 128 | iguais para todos os modelos |

**Leitura.**

1. **ALBERT:** o artigo usa learning rate 5× menor (1e-5) e warmup de ≈ 6% na tarefa mais parecida com a nossa. O protocolo deste trabalho (5e-5, sem warmup) fica fora dessa faixa, o que é coerente com o platô em ln 2 e com a instabilidade observada (Seções 6.4, 7.8 e 8.9). Dá base documentada ao experimento com warmup previsto na Seção 11.5.
2. **MiniLM:** o artigo recomenda 3 a 5 épocas; aqui são 10, o que é coerente com o sobreajuste observado (perda de avaliação mínima na 2ª época; Seção 8.9).
3. **MobileBERT:** 10 épocas e 5e-5 estão dentro da faixa do próprio artigo, coerente com o desempenho estável do modelo.
4. **TinyBERT:** o artigo não prevê fine-tuning direto do checkpoint geral — o procedimento original é a destilação específica de tarefa com o BERT como professor. Este trabalho (como, aparentemente, o PHILDER) faz fine-tuning direto. É uma diferença em relação ao TinyBERT original a ser registrada no artigo; ainda assim o modelo atinge o melhor resultado.
5. **Batch 8:** menor que o recomendado em todos os artigos (16 a 48).
6. **Consequência para a comparação:** o protocolo único, fiel ao PHILDER, pode desfavorecer o ALBERT e o MiniLM. Isso não altera a conclusão central (o TinyBERT, sob o mesmo protocolo, empata com o melhor modelo e é o mais eficiente), mas deve constar como **limitação** no artigo. Ajustar hiperparâmetros por modelo só é válido com um conjunto de validação separado do treino (Seção 11.5).

Fontes: model cards [43]–[47] e artigos [22], [24]–[28] da `bibliografia.md` (seção "Documentação dos checkpoints e hiperparâmetros de fine-tuning").
