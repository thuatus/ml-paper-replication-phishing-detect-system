# Embasamento Teórico e Base de Conhecimento — Métricas do PHILDER
### Replicação do PHILDER | Aplicações Distribuídas — PPGEE/UnB

> Este documento consolida a fundamentação teórica e estatística das métricas usadas no artigo BISPO, G. D. et al. *PHILDER: Lightweight Framework for Intelligent Phishing Detection on Resource-Limited Devices*. IEEE Access, v. 13, p. 131967–131979, 2025 — servindo de base de conhecimento para os slides 8–10 da Atividade 3 e para a seção de metodologia do artigo final da disciplina. Para o registro cronológico de todas as execuções do projeto (não só as relacionadas ao achado de vazamento de dados) e a comparação consolidada com o artigo original, ver `Analise_dos_Resultados.md`.

---

## 1. Contexto

A avaliação de modelos de machine learning para detecção de phishing tradicionalmente se apoia em **métricas de desempenho preditivo** (acurácia, precisão, recall, F1-score) — ou seja, métricas que respondem apenas a uma pergunta: *"o modelo acerta?"*.

O problema identificado pelos autores do PHILDER é que essa pergunta, sozinha, é insuficiente quando o objetivo final é rodar o modelo em **dispositivos com recursos computacionais limitados** (gateways de e-mail, IoT, sistemas embarcados). Um modelo pode ser extremamente preciso e, ainda assim, inviável na prática, porque:
- demora tempo demais para treinar (relevante quando o modelo precisa ser reajustado periodicamente);
- demora tempo demais para classificar um lote de mensagens (relevante em auditorias/testes em batch);
- demora tempo demais para classificar **uma única mensagem** em tempo real (relevante para um filtro de e-mail que precisa decidir na hora).

Por isso, o artigo propõe **duas famílias de métricas**, tratadas em conjunto:
1. **Métricas de desempenho preditivo** (clássicas, herdadas da literatura geral de classificação).
2. **Métricas compostas de eficiência computacional** — TES, IES e RTDE — que são a contribuição metodológica original do artigo, respondendo à pergunta *"a que custo o modelo acerta?"*.

---

## 2. Métricas de Desempenho Preditivo (clássicas)

Todas as métricas desta seção partem da **matriz de confusão** de um classificador binário (phishing = classe positiva; legítimo = classe negativa):

| | Previsto: Phishing | Previsto: Legítimo |
|---|---|---|
| **Real: Phishing** | Verdadeiro Positivo (VP) | Falso Negativo (FN) |
| **Real: Legítimo** | Falso Positivo (FP) | Verdadeiro Negativo (VN) |

### 2.1 Acurácia

**O que é:** a proporção geral de classificações corretas (positivas e negativas) sobre o total de exemplos avaliados.

**Como é calculada:**
$$\text{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN}$$

**Fundamento estatístico:** é, essencialmente, a taxa de sucesso empírica do classificador — um estimador pontual da probabilidade de acerto, calculado sobre a amostra de teste. Não pondera de forma diferente os dois tipos de erro (FP e FN), o que é exatamente sua limitação central.

**Limitação crítica para este projeto:** com um dataset desbalanceado (501 phishing vs. 2.501 legítimos no artigo original, ~83% de exemplos legítimos), um classificador "preguiçoso" que sempre prevê "legítimo" já atinge ~83% de acurácia sem detectar um único ataque. Isso torna a acurácia isolada uma métrica **enganosa** neste domínio — motivo pelo qual o artigo (e a literatura de classificação desbalanceada em geral) exige métricas complementares.

### 2.2 Precisão (Precision)

**O que é:** dentre tudo que o modelo classificou como phishing, qual fração realmente era phishing.

**Como é calculada:**
$$\text{Precisão} = \frac{VP}{VP + FP}$$

**Fundamento estatístico:** é uma probabilidade condicional — $P(\text{real é phishing} \mid \text{previsto como phishing})$. Mede o custo de **falsos alarmes**: precisão baixa significa que e-mails legítimos estão sendo indevidamente barrados (custo operacional para o usuário).

### 2.3 Recall (Sensibilidade / Revocação)

**O que é:** dentre todos os phishings reais, qual fração o modelo conseguiu capturar.

**Como é calculada:**
$$\text{Recall} = \frac{VP}{VP + FN}$$

**Fundamento estatístico:** também é uma probabilidade condicional — $P(\text{previsto como phishing} \mid \text{real é phishing})$. Mede o custo de **omissão**: recall baixo significa que ataques reais estão passando despercebidos, o erro mais custoso em segurança (um FN é um phishing que chega à caixa de entrada do usuário).

### 2.4 F1-score

**O que é:** uma métrica única que resume o equilíbrio entre precisão e recall, evitando que o modelo "otimize" uma às custas da outra.

**Como é calculada** (média harmônica, não aritmética):
$$F_1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}$$

**Por que média harmônica, e não aritmética (fundamento estatístico):** a média harmônica penaliza fortemente valores baixos em qualquer um dos dois termos. Se precisão = 1,0 e recall = 0,0 (um modelo que nunca marca nada como phishing), a média aritmética resultaria em 0,50 — um número enganosamente razoável. A média harmônica resulta em 0,0, refletindo corretamente que o modelo é inútil. É essa propriedade matemática que torna o F1 uma métrica muito mais informativa do que a acurácia isolada em cenários desbalanceados como este.

---

## 3. Métricas Compostas de Eficiência Computacional (contribuição central do PHILDER)

### 3.1 Fundamento estatístico comum: razão custo-benefício

As três métricas seguem a mesma lógica estrutural — uma **razão (taxa)** entre um indicador de desempenho (acurácia) e um indicador de custo (tempo, em segundos ou milissegundos):

$$\text{Métrica de eficiência} = \frac{\text{Acurácia}}{\text{Tempo}}$$

Do ponto de vista estatístico, isso é análogo a métricas de "produtividade" ou "retorno por unidade de recurso" usadas em outras áreas (ex.: FLOPS por watt em hardware, retorno por unidade de risco em finanças). A interpretação é sempre a mesma: **quanto maior o valor, mais "eficiente" é o modelo** — ele entrega mais acurácia por unidade de tempo gasto.

Uma implicação matemática importante: como o tempo está no denominador, a métrica é extremamente sensível a variações pequenas de tempo quando o tempo já é pequeno (situação típica de modelos leves como o TinyBERT) — por isso a recomendação, discutida no Slide 10 deste projeto, de rodar múltiplas execuções e reportar média ± desvio-padrão, já que ruído de sistema operacional no cronômetro pode distorcer a métrica de forma desproporcional.

### 3.2 TES — Training Efficiency Score

**O que é:** mede o quão "barato" foi, em tempo, chegar à acurácia obtida através do processo de treino (fine-tuning).

**Como é calculada:**
$$\text{TES} = \frac{\text{Acurácia}}{\text{Tempo de treino}}$$

**Para que serve na prática:** é relevante em cenários onde o modelo precisa ser **retreinado com frequência** — por exemplo, para incorporar novos padrões de phishing periodicamente. Um TES alto indica que o modelo pode ser reajustado rapidamente sem sacrificar muito a acurácia.

### 3.3 IES — Inference Efficiency Score

**O que é:** mede o custo de rodar o modelo já treinado sobre **todo o conjunto de teste** (inferência em lote/batch).

**Como é calculada:**
$$\text{IES} = \frac{\text{Acurácia}}{\text{Tempo de teste}}$$

**Para que serve na prática:** relevante em cenários de **auditoria em lote** — por exemplo, escanear uma caixa de entrada inteira ou uma base histórica de e-mails de uma só vez. Um IES alto indica que o modelo processa grandes volumes de mensagens rapidamente sem perder acurácia.

### 3.4 RTDE — Real-Time Detection Efficiency

**O que é:** mede o custo de classificar **uma única amostra** (um e-mail por vez) — a métrica mais diretamente ligada à viabilidade de uso em produção.

**Como é calculada:**
$$\text{RTDE} = \frac{\text{Acurácia}}{\text{Tempo de detecção por amostra}}$$

**Para que serve na prática:** é a métrica mais relevante para um **filtro de e-mail em tempo real** ou um gateway de segurança — o cenário-fim que justifica a existência do PHILDER. Diferente do IES (que mede throughput em lote), o RTDE mede **latência unitária**, que é o que realmente importa quando o usuário está esperando a resposta do sistema (ex.: um e-mail chegando e precisando ser classificado antes de aparecer na caixa de entrada).

### 3.5 Observação crítica: os valores de TES/RTDE reportados na Tabela 4 do artigo não são a razão bruta — e IES não recebe o mesmo tratamento estatístico

Ao cruzar os números que o próprio artigo publica em lugares diferentes — os gráficos de barras individuais de TES/IES/RTDE (valores brutos, por execução) e a **Tabela 4** do artigo (resultados consolidados) — fica claro que a Tabela 4 usa uma escala diferente da fórmula bruta definida na Seção 3.1, e que essa diferença não é arredondamento nem ruído de execução.

**O que a Tabela 4 realmente é.** O artigo declara explicitamente: *"statistical analysis was conducted across three independent runs for each model using the undersampling approach. Table 4 summarizes the mean and standard deviation for key performance metrics (Accuracy, F1-score, TES, and RTDE)"*. Ou seja, a Tabela 4 reporta **média ± desvio-padrão de 3 execuções independentes**, mas **apenas para 4 métricas**: Acurácia, F1, TES e RTDE. **IES fica de fora dessa tabela** — não há, em nenhum outro ponto identificado do artigo, uma consolidação estatística equivalente para IES.

**Evidência 1 — os gráficos de barras (valor bruto, execução única) são consistentes com a fórmula da Seção 3.1.** O gráfico "Training Efficiency (TES)" mostra TES ≈ 0,0389 para o PHILDER (TinyBERT + undersampling). Usando a acurácia do artigo (99,27%) e isolando o tempo de treino:

$$\text{Tempo de treino} = \frac{0{,}9927}{0{,}0389} \approx 25{,}5\ \text{s}$$

— plausível para o fine-tuning de um TinyBERT (4 camadas, 312 dimensões) num dataset pequeno. Da mesma forma, a latência de detecção reportada pelo artigo (12,98 ms/amostra) reproduz, pela fórmula bruta de RTDE, ≈ 76,5 — valor que bate com o RTDE calculado de forma totalmente independente a partir da tabela de resultados experimentais dos alunos (linha "TinyBERT (2)": ≈ 76,47). Os **gráficos de barras usam, portanto, a fórmula bruta (Acurácia/Tempo) tal como definida na Seção 3.1**.

**Evidência 2 — nem TES nem RTDE da Tabela 4 se explicam por uma normalização simples (valor bruto ÷ máximo do grupo).** Testando a hipótese mais direta de conversão de escala — dividir o valor bruto de cada um dos 5 modelos (todos com undersampling) pelo valor bruto máximo do grupo (que é sempre o do TinyBERT/PHILDER) — o resultado não bate com a Tabela 4 em nenhuma das duas métricas:

| Modelo | TES bruto | Bruto ÷ máximo | **TES — Tabela 4** |
|---|---|---|---|
| TinyBERT | 0,0389 | 1,000 | 0,92 ± 0,01 |
| MiniLM | 0,0231 | 0,594 | 0,91 ± 0,01 |
| MobileBERT | 0,0098 | 0,252 | 0,86 ± 0,02 |
| DistilBERT | 0,0089 | 0,229 | 0,89 ± 0,02 |
| ALBERT | 0,0042 | 0,108 | 0,82 ± 0,02 |

| Modelo | RTDE bruto | Bruto ÷ máximo | **RTDE — Tabela 4** |
|---|---|---|---|
| TinyBERT | 76,48 | 1,000 | 0,93 ± 0,01 |
| DistilBERT | 66,76 | 0,873 | 0,87 ± 0,03 |
| MiniLM | 61,23 | 0,801 | 0,91 ± 0,02 |
| MobileBERT | 44,00 | 0,575 | 0,85 ± 0,01 |
| ALBERT | 30,28 | 0,396 | 0,80 ± 0,02 |

Em ambas as métricas, a normalização "bruto ÷ máximo" produz uma faixa larga (0,11–1,00 no TES; 0,40–1,00 no RTDE), enquanto a Tabela 4 mostra uma faixa estreita e consistentemente alta (0,80–0,93 nas duas métricas, para os dois lados). O erro é maior justamente nos modelos mais lentos (ALBERT, MobileBERT) — o padrão oposto do que se esperaria de arredondamento ou ruído de medição. **A mesma normalização testada com IES (que não tem contraparte na Tabela 4 para comparar) produz uma faixa ainda mais larga — 0,09 a 1,00 — reforçando que essa métrica sequer foi levada à mesma forma de consolidação.**

**Leitura mais provável.** A Tabela 4 não é apenas "os valores brutos normalizados" — ela parece empregar uma fórmula, transformação de escala ou conjunto de referência diferente dos que produzem os gráficos de barras, e essa fórmula não é descrita explicitamente no texto disponível. É notável que os valores de TES e RTDE na Tabela 4 caiam numa faixa parecida com a de Acurácia/F1 (também na casa de 0,80–0,99) — o que sugere (sem confirmar) que a intenção editorial foi trazer as métricas de eficiência para uma escala comparável às métricas preditivas, mas o método usado para isso não está documentado.

**Status desta observação:** hipótese fundamentada em evidência numérica cruzada (múltiplos cálculos independentes convergindo para os mesmos valores brutos, e testes de normalização simples que falham de forma sistemática), mas **não é uma confirmação** de qual transformação foi de fato usada. Fica registrada como uma **lacuna de reprodutibilidade metodológica dupla**: (a) a Tabela 4 usa uma escala para TES/RTDE que não é derivável dos dados brutos publicados pelo próprio artigo por nenhum método simples testado; (b) o IES — uma das três métricas de eficiência apresentadas como contribuição central do artigo — não recebe nenhum tratamento estatístico consolidado (nem na Tabela 4, nem em outro lugar identificado). Ambos os pontos são relevantes para a entrega de "análise crítica dos métodos de aferição de desempenho do modelo" prevista na Fase 2 deste projeto: ao reportar TES/IES/RTDE na replicação de vocês, recomenda-se apresentar os valores **brutos** (fórmula da Seção 3.1, diretamente comparáveis entre execuções próprias), acompanhados de média ± desvio-padrão para as três métricas (não só duas) — e, se desejado, também uma versão normalizada, mas documentando explicitamente o método e o conjunto de referência usados, o que o artigo original não faz.

---

## 4. Para que servem os indicadores, em conjunto — o que eles medem na prática

| Métrica | Pergunta que responde | Cenário de uso onde é decisiva |
|---|---|---|
| Acurácia | O modelo acerta, no geral? | Visão geral rápida, mas enganosa se usada isoladamente |
| Precisão | Meus alarmes são confiáveis? | Evitar bloquear e-mails legítimos por engano (custo de usabilidade) |
| Recall | Estou pegando os ataques de verdade? | Evitar que phishing passe despercebido (custo de segurança) |
| F1-score | Há equilíbrio entre os dois erros? | Comparação justa entre modelos em dataset desbalanceado |
| TES | Vale a pena retreinar este modelo com frequência? | Manutenção contínua do sistema (re-treino periódico) |
| IES | O modelo processa grandes volumes rapidamente? | Auditoria/varredura em lote de uma base de e-mails |
| RTDE | O modelo responde rápido o suficiente para uso ao vivo? | Filtro de e-mail em tempo real, gateway de segurança, dispositivo edge |

Em conjunto, essas sete métricas formam um **retrato completo** do modelo: as quatro primeiras dizem se ele é *bom*; as três últimas dizem se ele é *viável* no hardware-alvo. É essa combinação, e não uma métrica isolada, que sustenta a escolha final do PHILDER.

---

## 5. Análise dos resultados relatados no artigo

### 5.1 O resultado vencedor
O artigo relata que a configuração **TinyBERT + undersampling** — que os autores batizam de **PHILDER** — obteve:
- **Acurácia de 99,27%**, superando um trabalho anterior de referência em NLP para detecção de phishing (98,20%);
- **Maior TES e maior IES** entre as 15 combinações testadas (5 modelos × 3 estratégias de balanceamento);
- **Latência de detecção (RTDE) de 12,98 ms por amostra** — ordem de grandeza compatível com uso em tempo real.

Vale registrar que, entre os 5 modelos testados apenas com undersampling (Tabela 4 do artigo), o TinyBERT **não** tem a maior acurácia nem o maior F1 isolados — o DistilBERT (undersampling) reporta Acurácia 0,9934 ± 0,0005 e F1 0,9932 ± 0,0006, ambos levemente acima do TinyBERT (0,9927 ± 0,0006 / 0,9927 ± 0,0004). O TinyBERT vence porque tem o maior TES (0,92) e o maior RTDE (0,93) da tabela — reforçando, com uma segunda evidência independente da do SMOTE (seção 5.2), que a escolha final do PHILDER é guiada pelas métricas de eficiência, não pela acurácia isolada.

### 5.2 Por que o SMOTE (oversampling) foi descartado, apesar de melhor acurácia isolada
Esse é o ponto mais didático do artigo para ilustrar o valor das métricas compostas: o SMOTE obteve as **melhores métricas preditivas isoladas** (acurácia e F1 mais altos), mas foi descartado porque:
1. Gera um dataset sintético consideravelmente maior (expande a classe minoritária de 501 para 2.501 exemplos), aumentando o tempo de treino;
2. Isso reduz o TES e o IES de forma acentuada — o ganho marginal de acurácia não compensa o custo computacional adicional;
3. Há também um risco reconhecido de overfitting a padrões sintéticos, e menor robustez observada nas métricas compostas.

**Leitura estatística desse resultado:** é um exemplo concreto de que **otimizar uma métrica isolada (acurácia) pode levar a uma decisão subótima** quando o critério de decisão real é multidimensional (acurácia + custo). É exatamente esse tipo de armadilha que as métricas TES/IES/RTDE foram desenhadas para expor.

### 5.3 O estudo de ablação e o rigor estatístico do artigo

Os autores conduziram um estudo de ablação isolando duas contribuições: (a) o uso de pesos pré-treinados (transfer learning) e (b) a técnica de undersampling. O resultado confirma que **ambos contribuem de forma significativa e independente** para o desempenho final — ou seja, nenhum dos dois fatores, sozinho, explica o resultado; a combinação dos dois é o que sustenta o desempenho do PHILDER.

**Correção em relação a uma leitura anterior deste documento:** o artigo **não** está totalmente isento de rigor estatístico, como uma versão anterior desta seção sugeria. A Tabela 4 do artigo relata explicitamente **média ± desvio-padrão de 3 execuções independentes** para Acurácia, F1-score, TES e RTDE, na comparação entre os 5 modelos com undersampling. A lacuna real é mais específica do que "ausência de rigor estatístico": (a) esse tratamento estatístico (3 execuções, média ± desvio-padrão) não é estendido ao **IES**, a Precisão nem ao Recall — apenas às 4 métricas citadas; (b) o estudo de ablação em si (pesos pré-treinados × undersampling) não é acompanhado, no texto disponível, de nenhuma medida de variabilidade entre execuções; (c) como discutido na Seção 3.5, mesmo as métricas que recebem tratamento estatístico (TES, RTDE) são reportadas na Tabela 4 numa escala que não corresponde à fórmula bruta definida na própria metodologia do artigo, o que é uma lacuna de **reprodutibilidade** distinta da lacuna de rigor estatístico. Essas três lacunas específicas (IES sem consolidação estatística, ablação sem variabilidade reportada, e escala de TES/RTDE não documentada) são o que a replicação de vocês pode suprir de forma concreta, rodando cada configuração múltiplas vezes com sementes diferentes e reportando média ± desvio-padrão para as sete métricas, sempre na escala bruta e documentada.

### 5.4 MiniLM como alternativa competitiva
O artigo aponta o **MiniLM com undersampling** como segunda melhor combinação — mais leve que o TinyBERT, mas com desempenho preditivo ligeiramente inferior. Isso reforça que a escolha do TinyBERT não foi por ausência de alternativas, mas por representar o melhor ponto no trade-off acurácia × eficiência entre as cinco variantes testadas.

### 5.5 Achado metodológico da replicação: vazamento de dados (data leakage) entre treino e teste, e a evolução da correção em duas etapas

> **Nota sobre esta seção (atualizada em 25/09):** o diagnóstico abaixo cobre a primeira rodada de investigação (21–23/09) e documenta como o vazamento de dados foi detectado e uma primeira correção foi desenhada. **Importante:** uma segunda rodada de investigação, em 24–25/09 e descrita na Seção 5.6, descobriu que a causa raiz real dos resultados "bons demais" não estava na lógica de deduplicação em si (que sempre esteve matematicamente correta), mas em dois bugs de implementação no notebook — um deles fazia com que a célula do Módulo A nunca chegasse a chamar `amostrar_quantidade_artigo`, isto é, a "Versão 2 (23/09)" descrita abaixo **não estava, de fato, em execução** no notebook naquela data, apesar do código do módulo `.py` estar correto. Por isso, os números de 21/09–23/09 permanecem registrados aqui como o histórico do processo de detecção do vazamento (didaticamente valiosos), mas **não devem mais ser tratados como o resultado de referência da replicação** — esse papel passa a ser do resultado de 25/09 (Seção 5.6), o primeiro cuja cadeia de execução foi de fato verificada linha a código e confirmada por uma checagem explícita de sobreposição treino/teste. A linha do tempo completa de todas as execuções do projeto está consolidada em `Analise_dos_Resultados.md`.

Durante a replicação, um primeiro treinamento — já na escala de dataset do artigo (2.501 legítimos + 501 phishing, via `amostrar_quantidade_artigo`) e com o regime de 10 épocas + *early stopping* descrito na Seção 7 — produziu um resultado suspeito: **Precisão exatamente 1,0** (Acurácia 0,9933; Recall 0,9867; F1 0,9933). Uma precisão perfeita é um sinal de alerta clássico em ML, especialmente após uma mudança que reduz o tamanho do dataset (aumenta a chance relativa de repetição de conteúdo).

**Diagnóstico.** A matriz de confusão reconstruída a partir dessas métricas (teste com 301 amostras: 150 phishing + 151 legítimos) indicava VP=148, FN=2, FP=**0**, VN=151 — zero falsos positivos é o padrão que gera precisão=1,0. Isso levou à checagem de duplicidade no dataset:
- `df["body"].duplicated().sum()` → 30 duplicatas exatas no corpo bruto do e-mail;
- `df["body_processado"].duplicated().sum()` → 76 duplicatas após o pré-processamento (Módulo B), de 3.002 registros (16 deles esvaziados pela limpeza);
- o teste decisivo — sobreposição entre os conjuntos de treino e teste já divididos — confirmou **7 e-mails processados idênticos presentes simultaneamente no treino e no teste** (`set(df_treino["body_processado"]) & set(df_teste["body_processado"])`).

Ou seja: parte dos e-mails do conjunto de teste tinha uma cópia (quase) idêntica no conjunto de treino. O modelo não estava generalizando para esses casos — estava reconhecendo um exemplo que já tinha visto, o que infla artificialmente a precisão sem refletir capacidade real de detecção.

**De qual fonte vêm as duplicatas: SpamAssassin ou Nazario?** Cruzando a duplicidade com `class_label` (0 = legítimo/SpamAssassin, 1 = phishing/Nazario) no dataset de 3.002 registros (2.501 legítimos + 501 phishing, escala do artigo):

| Classe | Tamanho do pool | Registros envolvidos em duplicação | Duplicatas removidas (`drop_duplicates`) | Taxa de duplicação | Grupos distintos de conteúdo duplicado | Tamanho médio do grupo |
|---|---|---|---|---|---|---|
| 0 — Legítimo (SpamAssassin) | 2.501 | 86 | 56 | 2,24% | 30 | 2,87 |
| 1 — Phishing (Nazario) | 501 | 27 | 20 | 3,99% | 7 | 3,86 |
| **Total** | **3.002** | **113** | **76** | **2,53%** | **37** | **3,05** |

Em número absoluto, a maior parte das duplicatas removidas vem do **SpamAssassin** (56 de 76, ou 74%) — não do Nazario, ao contrário da hipótese inicial de que campanhas de phishing reutilizando o mesmo template seriam a fonte dominante. Isso se explica pelo tamanho dos pools: o SpamAssassin contribui com 5× mais registros (2.501 vs. 501), então mesmo com uma taxa de duplicação menor ele ainda gera mais duplicatas em números absolutos.

Olhando pela **taxa relativa** (duplicatas ÷ tamanho do pool), porém, o padrão se inverte: o Nazario tem uma taxa de duplicação quase o dobro da do SpamAssassin (3,99% vs. 2,24%) — proporcionalmente, o corpus de phishing é mais redundante. E olhando o **tamanho médio dos grupos de duplicatas** (quantas cópias cada conteúdo repetido tem), o padrão de phishing aparece de novo: os 7 grupos duplicados do Nazario têm, em média, 3,86 cópias cada, contra 2,87 no SpamAssassin — ou seja, o Nazario tem *menos* conteúdos distintos duplicados, mas cada um se repete *mais vezes*, consistente com a hipótese original de reaproveitamento de templates em campanhas de phishing (menos "famílias" de ataque, cada uma disparada em maior volume), enquanto no SpamAssassin a duplicação é mais difusa (mais grupos, cada um menor — provável efeito de e-mails automáticos/newsletters com corpo quase idêntico).

**Confirmação: a deduplicação explica exatamente a queda do tamanho do teste (301 → 289) na primeira correção.** Isolando os 20 registros de phishing removidos: o pool de phishing único cai de 501 para **481** (501 − 20); o de legítimos únicos cai de 2.501 para 2.445 (2.501 − 56). Como o balanceamento por undersampling (Módulo C) usa o tamanho da classe minoritária como alvo, o novo alvo passa a ser 481 (em vez de 501), gerando um dataset balanceado de 481 + 481 = **962** registros (em vez de 1.002). Aplicando a mesma proporção de divisão treino/teste inferida da execução de 21/09 (301/1.002 ≈ 30%): 962 × 0,30 ≈ **288,6 ≈ 289** — batendo exatamente com o tamanho de teste observado na execução de 22/09. Ou seja, a queda no tamanho do teste não foi um efeito indireto vago: foi consequência direta e quantificável de o Nazario (phishing) ter perdido 20 exemplares únicos na deduplicação, reduzindo o alvo do undersampling.

**Correção "desenhada" — duas versões (ver ressalva na nota do topo desta seção sobre a Versão 2 não ter estado, de fato, em execução no notebook nessa data).**

*Versão 1 (22/09).* Inserção de `df = df.drop_duplicates(subset="body_processado").reset_index(drop=True)` logo após o `pipeline_processamento` (Módulo B), sobre o dataset já consolidado (legítimos + phishing juntos), antes do balanceamento/divisão treino-teste (Módulo C). Resolve o vazamento, mas remove duplicatas dos dois lados (56 legítimos + 20 phishing) *depois* de já ter amostrado exatamente 501 e-mails de phishing — como o alvo do undersampling passa a ser o tamanho da classe minoritária *após* a deduplicação (481, não mais 501), o dataset balanceado encolhe de 1.002 para 962 registros, e o teste de 301 para 289 amostras (demonstração acima).

*Versão 2 (23/09) — correção desenhada, mas cuja execução efetiva no notebook não pôde ser confirmada (ver Seção 5.6).* Em vez de deduplicar depois de amostrar, o pool bruto de phishing (Nazario) passaria a ser deduplicado pelo texto processado *antes* da amostragem dos 501 e-mails — parâmetro `deduplicar_antes=True` (padrão) de `amostrar_quantidade_artigo`, Módulo A (`database_consolidation.py`). A lógica do módulo `.py` estava correta desde então, mas só em 24/09 se descobriu que a célula do notebook que orquestra o Módulo A nunca chegou a invocar essa função (Seção 5.6) — de modo que não é possível afirmar com certeza qual pipeline gerou, na prática, os números de 23/09 reportados abaixo.

**Resultado das três execuções**, mesmo regime de treino (10 épocas, *early stopping*) — números históricos, mantidos para registro do processo de diagnóstico (ver ressalva sobre 23/09 acima):

| | 21/09 — Antes (com vazamento) | 22/09 — Depois v1 (dedup pós-consolidação) | 23/09 — Depois v2 "desenhada" (pipeline não confirmado — ver Seção 5.6) |
|---|---|---|---|
| n (teste) | 301 | 289 | 301 |
| Acurácia | 0,9933 | 0,9792 | 0,9900 |
| Precisão | **1,0000** | 0,9799 | 0,9868 |
| Recall | 0,9867 | 0,9792 | **0,9933** |
| F1 | 0,9933 | 0,9792 | 0,9900 |
| Matriz (VP/FN/FP/VN) | 148 / 2 / 0 / 151 | ≈ 142 / 3 / 3 / 141* | 149 / 1 / 2 / 149 |
| Tempo de treino | 205,51 s | 298,14 s | 263,60 s |
| Tempo de teste | 2,159 s | 2,068 s | 2,307 s |
| Detecção/amostra | 7,173 ms | 7,156 ms | 7,663 ms |
| TES | ≈ 0,004834 | ≈ 0,003284 | ≈ 0,003756 |
| IES | ≈ 0,460120 | ≈ 0,473501 | ≈ 0,429184 |
| RTDE | ≈ 138,48 | ≈ 136,844 | ≈ 129,187 |

*matriz aproximada, reconstruída a partir das métricas arredondadas fornecidas (não das contagens brutas) — a pequena diferença entre precisão e recall nessa execução não permite fixar os quatro valores com exatidão, apenas estimar que os erros (6 no total, de 289 amostras) se distribuem de forma próxima a 3 FP / 3 FN. Nas colunas de 21/09 e 23/09, os quatro valores da matriz foram reconstruídos com exatidão a partir das métricas relatadas.

Ainda que o mecanismo exato por trás do resultado de 23/09 não possa mais ser afirmado com certeza (Seção 5.6), o padrão observado na época — precisão descendo de 1,0 para 0,9868 e recall subindo para o maior valor das três execuções (0,9933) — foi o que motivou a continuidade da investigação e, por isso, permanece registrado como parte do histórico. A Seção 5.6 apresenta o resultado que, esse sim, teve sua cadeia de execução completamente verificada.

**Relevância para a análise crítica da Fase 2.** Este achado é uma contribuição metodológica concreta da replicação, complementar às lacunas já identificadas no artigo original (Seções 3.5 e 5.3): o artigo PHILDER não menciona, em nenhum trecho disponível, uma etapa de deduplicação antes da divisão treino/teste — prática básica recomendada na literatura de ML para evitar vazamento de dados. Reportar apenas a métrica de precisão isolada (como a Tabela 4 do artigo faz para F1, mas sem expor a matriz de confusão) pode mascarar justamente esse tipo de problema. Além disso, o processo de correção em várias etapas — diagnosticar o vazamento, desenhar uma correção, descobrir um efeito colateral não previsto (encolhimento do dataset), corrigir esse efeito colateral e, só depois, descobrir que a própria célula do notebook nunca executava a correção desenhada (Seção 5.6) — é, em si, um exemplo instrutivo de por que resultado "razoável" não é o mesmo que "pipeline verificado": só a checagem explícita de sobreposição treino/teste (não a plausibilidade das métricas) fecha a questão com certeza. Recomenda-se, para o artigo final da disciplina: (a) documentar explicitamente a etapa de deduplicação como parte do pipeline de pré-processamento, incluindo a quebra por fonte de dados (SpamAssassin vs. Nazario) e o ponto do pipeline em que ela ocorre (antes vs. depois da amostragem); (b) relatar o número de duplicatas removidas e a sobreposição treino/teste antes e depois da correção, como evidência de rigor metodológico; (c) tratar precisão ou acurácia excepcionalmente altas (próximas de 1,0) como gatilho padrão para uma checagem de vazamento de dados, antes de aceitar o resultado como válido; (d) preferir, sempre que possível, deduplicar o pool bruto antes de reduzi-lo ao tamanho-alvo, em vez de deduplicar depois — evita o efeito colateral de encolhimento observado na Versão 1; (e) **nunca considerar um resultado "razoável" como prova de que o pipeline pretendido de fato executou** — a única evidência aceita como conclusiva, adotada a partir da Seção 5.6, é a checagem explícita e programática de `set(df_treino["body_processado"]) & set(df_teste["body_processado"])` retornando vazio, associada à inspeção literal do código efetivamente executado pela célula do notebook (não do módulo `.py` isoladamente).

### 5.6 Segunda rodada de investigação (24–25/09): a causa raiz real e o resultado de referência definitivo

Com a Versão 2 (23/09) adotada como referência provisória, o passo seguinte planejado era estender a mesma lógica de deduplicação prévia (`deduplicar_antes=True`) também ao lado dos legítimos (SpamAssassin), que até então não recebia tratamento prévio — item que já constava como "em aberto" na Seção 6 de `Analise_dos_Resultados.md`. Ao implementar essa extensão, em 24/09, o resultado voltou a apresentar precisão suspeita (próxima de 1,0), reabrindo a investigação.

**Bug 1 — a célula do notebook nunca chamava `amostrar_quantidade_artigo`.** A inspeção da célula real do notebook que orquestra o Módulo A revelou que ela nunca havia sido atualizada para usar a função de amostragem com deduplicação — em vez disso, mantinha um trecho residual de 19/09, anterior a toda a investigação de vazamento:

```python
# --- Módulo A: consolidação da base ---
brutos_legitimos = consolidacao.importar_corpus_spamassassin("dados_philder/spamassassin")
brutos_phishing = consolidacao.importar_corpus_nazario("dados_philder/Nazario.csv")

# 19/09/2026 - ...
random.seed(42)
brutos_phishing = random.sample(brutos_phishing, 501)

df = consolidacao.consolidar_dataset(brutos_legitimos, brutos_phishing)
```

Ou seja: todas as edições feitas no módulo `.py` entre 21/09 e 23/09 (Seção 5.5) eram código morto — nunca chegavam a ser executadas, porque a célula do notebook amostrava o phishing manualmente (sem deduplicação) e alimentava `consolidar_dataset` diretamente com as listas brutas. Isso explica por que a precisão continuava alta mesmo depois da lógica de deduplicação estar, em tese, implementada: ela simplesmente nunca rodava.

**Correção 1.** A célula foi reescrita para de fato chamar a função correta, passando as listas amostradas (não as brutas) para a consolidação:

```python
# --- Módulo A: consolidação da base ---
brutos_legitimos = consolidacao.importar_corpus_spamassassin("dados_philder/spamassassin")
brutos_phishing = consolidacao.importar_corpus_nazario("dados_philder/Nazario.csv")

legitimos_amostrados, phishing_amostrados = consolidacao.amostrar_quantidade_artigo(
    brutos_legitimos, brutos_phishing, deduplicar_antes=True
)

df = consolidacao.consolidar_dataset(legitimos_amostrados, phishing_amostrados)
```

Essa correção foi validada pelo log de execução (`"Amostragem para escala do artigo: 2446 legítimos + 501 phishing = 2947 e-mails ... Dataset consolidado: ... legítimos: 2446, phishing: 501"`), confirmando que a deduplicação prévia agora reduz o pool de legítimos únicos de 2.501 para 2.446 (55 duplicatas removidas — próximo às 56 identificadas na Seção 5.5) e mantém os 501 phishing (pool já deduplicado, sem excedente a remover no tamanho-alvo).

**Bug 2 — troca de variáveis ao estender a deduplicação para os legítimos.** Ao editar `amostrar_quantidade_artigo` para deduplicar também o pool de legítimos, uma segunda edição do usuário introduziu uma inversão: dentro do bloco `if deduplicar_antes:`, a chamada que deveria alimentar `legitimos_amostrados` recebia o pool de *phishing*, e vice-versa:

```python
# Versão com bug (rótulos de classe invertidos):
if deduplicar_antes:
    legitimos_amostrados = _amostrar_sem_duplicatas(
        emails_phishing_brutos, n_phishing, random_state, nome="phishing"
    )
    phishing_amostrados = _amostrar_sem_duplicatas(
        emails_legitimos_brutos, n_legitimos, random_state, nome="legítimos"
    )
```

O efeito prático: o dataset consolidado passou a ter as classes inteiramente trocadas (log mostrando `"501 legítimos + 2446 phishing"`, quando o esperado era o oposto). Esse bug foi diagnosticado a partir do padrão de números no próprio log de execução, antes mesmo de o código ser revisado, e confirmado ao comparar com a função colada na íntegra. A correção foi trocar qual pool alimenta qual variável:

```python
if deduplicar_antes:
    phishing_amostrados = _amostrar_sem_duplicatas(
        emails_phishing_brutos, n_phishing, random_state, nome="phishing"
    )
    legitimos_amostrados = _amostrar_sem_duplicatas(
        emails_legitimos_brutos, n_legitimos, random_state, nome="legítimos"
    )
```

Validada pelo log corrigido: `"Amostragem para escala do artigo: 2446 legítimos + 501 phishing = 2947 e-mails ... [deduplicar_antes=True]"`.

**Validação final, em três camadas independentes.**
1. **Log de execução** confirmando o tamanho e o rótulo corretos dos dois pools (2.446 legítimos + 501 phishing).
2. **Seis execuções repetidas** do pipeline completo (mesma configuração, "lote 25/09"), mostrando variância saudável — entre 0 e 2 erros por execução, em 301 amostras de teste — e, pela primeira vez desde o início da investigação, falsos positivos aparecendo do lado dos **legítimos** (não só falsos negativos do lado do phishing), o que é evidência adicional contra vazamento sistemático (um vazamento tende a produzir sempre o mesmo padrão de erro zero, não erros nos dois sentidos).
3. **Checagem programática de sobreposição treino/teste**, a prova definitiva:

```python
df_balanceado = balanceamento.aplicar_undersampling(df)
df_treino, df_teste = balanceamento.dividir_treino_teste(df_balanceado)

dup_no_balanceado = df_balanceado.duplicated(subset="body_processado", keep=False)
print("duplicatas no balanceado:", dup_no_balanceado.sum(), "de", len(df_balanceado))
print(df_balanceado["class_label"].value_counts())

dup_legit = df_balanceado[df_balanceado["class_label"] == 0].duplicated(subset="body_processado", keep=False)
dup_phish = df_balanceado[df_balanceado["class_label"] == 1].duplicated(subset="body_processado", keep=False)
print("duplicatas DENTRO de legítimos:", dup_legit.sum())
print("duplicatas DENTRO de phishing:", dup_phish.sum())

sobreposicao = set(df_treino["body_processado"]) & set(df_teste["body_processado"])
print(f"E-mails processados idênticos entre treino e teste: {len(sobreposicao)}")
```

Resultado obtido:

```
Balanceamento aplicado: {1: 501, 0: 501}
Treino: {1: 351, 0: 350}
Teste: {0: 151, 1: 150}
duplicatas no balanceado: 0 de 1002
duplicatas DENTRO de legítimos: 0
duplicatas DENTRO de phishing: 0
E-mails processados idênticos entre treino e teste: 0
```

Todos os quatro indicadores em zero — este é o primeiro momento da replicação em que a ausência de vazamento foi confirmada de forma direta e programática, e não apenas inferida da plausibilidade das métricas preditivas.

**Resultado de referência definitivo (25/09) — lote de 6 execuções, ambos os pools deduplicados antes da amostragem, sobreposição treino/teste = 0:**

| Execução | Acc. | Prec. | Rec. | F1 | T. Treino (s) | T. Teste (s) | Detecção (ms) | VP/FN/FP/VN |
|---|---|---|---|---|---|---|---|---|
| 1 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 389,18 | 4,38 | 14,55 | 149/1/0/151 |
| 2 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 343,57 | 3,98 | 13,21 | 150/0/0/151 |
| 3 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 275,64 | 3,85 | 12,79 | 149/1/0/151 |
| 4 | 0,9967 | 1,0000 | 0,9933 | 0,9967 | 344,83 | 4,06 | 13,49 | 149/1/0/151 |
| 5 | 0,9934 | 0,9933 | 0,9933 | 0,9933 | 355,99 | 3,71 | 12,33 | 149/1/1/150 |
| 6 | 0,9967 | 0,9934 | 1,0000 | 0,9967 | 404,24 | 3,55 | 11,79 | 150/0/1/150 |
| **Média ± DP** | **0,9967 ± 0,0021** | **0,9978 ± 0,0034** | **0,9956 ± 0,0034** | **0,9967 ± 0,0021** | **352,24 ± 44,90** | **3,92 ± 0,29** | **13,03 ± 0,96** | — |

Métricas de eficiência (TES, IES, RTDE), calculadas execução a execução pela fórmula bruta (Seção 3.1) e depois consolidadas em média ± desvio-padrão — a prática que a Seção 3.5 recomenda e que o artigo original não aplica ao IES:

| Métrica | Média ± DP (6 execuções) |
|---|---|
| TES | 0,002872 ± 0,000406 |
| IES | 0,255279 ± 0,018430 |
| RTDE | 76,8516 ± 5,5475 |

Pela primeira vez, erros aparecem dos dois lados da matriz de confusão (FP=1 nas execuções 5 e 6, além dos FN habituais), e uma das seis execuções (execução 2) atinge acurácia/precisão/recall/F1 perfeitos — um padrão de variância consistente com ausência de vazamento, corroborando a checagem programática acima. Este lote de 25/09 substitui o de 23/09 como **resultado de referência oficial da replicação** a partir daqui, inclusive nas comparações da Seção 5 de `Analise_dos_Resultados.md`.

**Lição metodológica para a Fase 2.** A causa raiz definitiva dos "resultados bons demais demais" recorrentes nunca esteve na estatística ou na lógica de deduplicação (`_amostrar_sem_duplicatas` esteve correta em todas as versões) — esteve em dois bugs de "última milha" muito comuns em pipelines de notebook: (1) uma célula desatualizada que nunca foi religada à nova lógica implementada no módulo, e (2) uma inversão de variáveis ao estender uma função existente. Nenhum dos dois seria detectável só olhando as métricas finais parecerem plausíveis — só ficaram evidentes ao (a) inspecionar a célula do notebook realmente executada, não só o módulo `.py`, e (b) rodar uma checagem programática explícita de sobreposição treino/teste e de rótulos de classe, em vez de confiar na plausibilidade do resultado. Essa é, em si, uma recomendação concreta e transferível para a análise crítica da Fase 2: **um resultado "razoável" não é evidência de que o pipeline pretendido executou** — só a checagem direta do código executado e dos dados resultantes é.

---

## 6. Síntese para uso nos slides

Ao levar este conteúdo para os slides 8–10 da apresentação, a sequência lógica recomendada é:
1. Apresentar as métricas clássicas (Slide 10) explicando **por que** acurácia isolada é insuficiente aqui (dataset desbalanceado).
2. Apresentar as métricas compostas (TES/IES/RTDE) como resposta direta a essa limitação, sempre reforçando a ideia de "acurácia por unidade de custo".
3. Usar o caso do SMOTE descartado (seção 5.2 acima) como o exemplo mais didático de por que essas métricas importam na prática — é um resultado concreto e contra-intuitivo do próprio artigo.
4. Fechar com as lacunas de rigor estatístico e de reprodutibilidade identificadas (seções 3.5 e 5.3) — e, se houver espaço, o achado de vazamento de dados e sua investigação em duas rodadas (seções 5.5 e 5.6), incluindo a diferença de comportamento entre SpamAssassin e Nazario e a lição de que só a checagem programática de sobreposição treino/teste confirma a ausência de vazamento — como gancho para a contribuição de vocês.

---

## 7. Padrões de reprodutibilidade para publicação do repositório

Ao publicar este projeto como repositório pessoal (Git/GitHub), existem convenções já estabelecidas na comunidade de ML/pesquisa reprodutível que se aplicam diretamente ao que já vem sendo documentado aqui:

| Padrão | O que exige | Relação com este projeto |
|---|---|---|
| **ML Reproducibility Checklist** (Pineau et al., NeurIPS 2019) | Código completo, dados (ou como obtê-los), todos os hiperparâmetros, nº de execuções/sementes, média ± desvio-padrão entre execuções, hardware usado | É o que já orienta as Seções 3.5 e 5.3 deste documento (lacunas de rigor estatístico do artigo original) — seguir a checklist na replicação é o contraponto direto a essas lacunas |
| **ACM Artifact Review and Badging** | Selos formais: *Artifacts Available* (código/dados publicados), *Artifacts Evaluated – Functional/Reusable* (roda e é reutilizável), *Results Reproduced* (terceiros bateram os mesmos resultados) | Não é avaliação formal aqui, mas a mesma lógica organiza o que declarar no README: o que está disponível, o que funciona, o que foi reproduzido com sucesso vs. divergiu |
| **Cookiecutter Data Science** | Estrutura de diretórios padronizada (`data/`, `notebooks/`, pacote de código, `models/`, `reports/`) | Compatível com a estrutura já adotada (`dados_philder/`, `philder/`, `modelo/`, `resultados/`) — não exige reorganizar, só documentar a correspondência |
| **Convenção de README para replicação de artigo** | Citação do artigo original; seção explícita de **divergências metodológicas**; `requirements.txt`/`environment.yml` com versões fixas; comando(s) de reprodução; tabela de resultados obtidos vs. artigo; `CITATION.cff` (opcional) | As divergências já documentadas nos guias (Nazario no lugar do PhishTank, ordem do undersampling, escala do dataset) são exatamente o conteúdo dessa seção — só precisam ser consolidadas no README |

**Estrutura de README recomendada, juntando os quatro:** citação do artigo → objetivo da replicação → estrutura do repositório → instalação (dependências fixadas) → como reproduzir → divergências metodológicas documentadas (fonte de dados, escala do dataset, ordem do undersampling, deduplicação treino/teste — Seções 5.5 e 5.6) → resultados obtidos vs. artigo (tabela consolidada em `Analise_dos_Resultados.md`, incluindo as lacunas de reprodutibilidade das Seções 3.5/5.3 deste documento) → licença.

---

## Referências

BISPO, G. D. et al. *PHILDER: Lightweight Framework for Intelligent Phishing Detection on Resource-Limited Devices*. IEEE Access, v. 13, p. 131967–131979, 2025.

DEVLIN, J.; CHANG, M.-W.; LEE, K.; TOUTANOVA, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805, 2018/2019.

JIAO, X.; YIN, Y.; SHANG, L.; JIANG, X.; CHEN, X.; LI, L.; WANG, F.; LIU, Q. *TinyBERT: Distilling BERT for Natural Language Understanding*. arXiv:1909.10351, 2019/2020.

PINEAU, J. et al. *Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)*. Journal of Machine Learning Research, v. 22, 2021. Disponível em: https://arxiv.org/abs/2003.12206

ASSOCIATION FOR COMPUTING MACHINERY. *Artifact Review and Badging — Current*. Disponível em: https://www.acm.org/publications/policies/artifact-review-and-badging-current

DRIVENDATA. *Cookiecutter Data Science*. Disponível em: https://cookiecutter-data-science.drivendata.org/
