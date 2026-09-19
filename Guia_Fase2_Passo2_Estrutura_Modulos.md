# Fase 2 — Passo 2: Estrutura em Módulos (espelhando a Parte III do artigo) + Autoreload
### Replicação fiel do PHILDER | Aplicações Distribuídas

> Segunda etapa da Fase 2. Substitui o padrão de "tudo em células soltas do notebook" por um pacote Python reutilizável (`philder/`), importado pelo notebook e, futuramente, também pelo script standalone de classificação em Linux e pela imagem de inferência do container (`Infra-Container.md`) — evitando duplicar a mesma lógica em lugares diferentes.
>
> **Revisão desta versão:** a organização interna do pacote foi reestruturada para espelhar diretamente os "capítulos" da **Parte III (Metodologia)** do artigo PHILDER, em vez de seguir a divisão por Passo da Fase 1. Agora são 4 módulos (A, B, C, D), cada um correspondendo a uma etapa da metodologia descrita no artigo. As funções seguem como **esqueleto** (assinatura + docstring, sem lógica interna) — a implementação entra depois da releitura do artigo.
>
> **Nota sobre nomes de arquivo:** os nomes foram gravados com a grafia corrigida (`database_consolidation.py`, `database_balance.py`, `performance_assessment.py`) em vez da grafia enviada originalmente (`cosolidation`, `balamce`, `assesment`). Se preferir manter os nomes exatamente como enviados, é só avisar.

---

## 1. Estrutura de diretórios atualizada

```
projeto_philder/
├── dados_philder/
├── modelo/
│   └── checkpoints/
├── resultados/
├── philder/                          # <- pacote reutilizável
│   ├── __init__.py
│   ├── database_consolidation.py     # Módulo A — Parte III: fontes de dados + rótulo
│   ├── data_processing.py            # Módulo B — Parte III: limpeza/normalização de texto
│   ├── database_balance.py           # Módulo C — Parte III: balanceamento, tokenização, treino, teste
│   └── performance_assessment.py     # Módulo D — Parte III: métricas + curva de perda
├── preparar_estrutura.py
└── test_estrutura_diretorios.py
```

`philder/` fica na raiz do projeto, ao lado do notebook — assim `from philder import database_consolidation` funciona sem nenhuma configuração extra de path.

---

## 2. Mapeamento módulo ↔ etapa da Parte III do artigo

| Módulo | Arquivo | Etapa da metodologia (Parte III) | Funções |
|---|---|---|---|
| **A** | `database_consolidation.py` | Import das fontes de dados + criação do rótulo de classificação | `importar_corpus_spamassassin`, `importar_corpus_nazario`, `extrair_campos_email`, `criar_rotulo_classificacao`, `consolidar_dataset`, `carregar_dataset` |
| **B** | `data_processing.py` | Remoção de cabeçalho SMTP, substituição de URLs, normalização de texto, remoção de stopwords | `remover_cabecalho_smtp`, `substituir_urls`, `normalizar_texto`, `remover_stopwords`, `pipeline_processamento` |
| **C** | `database_balance.py` | Undersampling, tokenização (BERT padrão), split 70/30, treino e teste do modelo | `aplicar_undersampling`, `dividir_treino_teste`, `carregar_tokenizer_e_modelo`, `tokenizar_dataset`, `treinar_modelo`, `testar_modelo`, `salvar_modelo`, `carregar_modelo_treinado` |
| **D** | `performance_assessment.py` | Métricas primárias, métricas de eficiência, curva de perda | `calcular_metricas_primarias`, `calcular_metricas_eficiencia`, `gerar_curva_perda`, `salvar_resultados` |

### Schema de dados que passa a valer a partir do Módulo A

Diferente da Fase 1 (colunas simples `texto`/`rotulo`), a consolidação agora produz um registro por e-mail com os campos discretos que o artigo utiliza:

```
sender, subject, body, smtp_headers, class_label
```

O `body` é a coluna que percorre os módulos B e C (`body` → `body_processado` após o Módulo B), enquanto `sender`, `subject` e `smtp_headers` ficam disponíveis para eventual uso futuro (ex.: features adicionais, ou apenas para preservar a fidelidade ao dado bruto). Essa mudança de schema é a principal diferença estrutural em relação ao pacote de 5 módulos da versão anterior deste guia — `dados.py`, `preprocessamento.py`, `balanceamento.py`, `modelo.py` e `avaliacao.py` foram removidos e sua lógica redistribuída entre A/B/C/D.

Cada função ainda está com `raise NotImplementedError("Implementar após revisão do artigo (Fase 2).")` no corpo — proposital: torna explícito, se você tentar rodar algo agora, exatamente o que falta preencher.

**Por que `database_balance.carregar_tokenizer_e_modelo` e `database_balance.carregar_modelo_treinado` ficam junto do Módulo C, e não separadas em um módulo próprio:** o artigo não trata "carregar modelo para inferência" como uma etapa própria da metodologia (isso é uma necessidade de engenharia, não do artigo) — por isso permanecem dentro do módulo de treino/teste. Ainda assim, seguem isoladas das demais funções de C: são as únicas que o futuro script standalone de classificação em Linux e a imagem de inferência enxuta do container (`Infra-Container.md`) vão precisar importar, sem precisar de `database_consolidation.py`, `data_processing.py` (exceto o pipeline de limpeza, que a inferência também precisa) nem das funções de treino propriamente ditas.

---

## 3. Configurando o `autoreload` no notebook

Na **primeira célula** do notebook, antes de qualquer `import` dos módulos do `philder`:

```python
%load_ext autoreload
%autoreload 2
```

Isso garante que qualquer alteração salva num arquivo `.py` do pacote seja recarregada automaticamente na próxima execução de célula — sem precisar reimportar manualmente ou reiniciar o kernel a cada ajuste de código.

---

## 4. Como o notebook passa a ficar (esqueleto do fluxo principal)

```python
%load_ext autoreload
%autoreload 2

from philder import database_consolidation as consolidacao
from philder import data_processing as processamento
from philder import database_balance as balanceamento
from philder import performance_assessment as avaliacao
from preparar_estrutura import criar_estrutura_diretorios

# --- Preparação ---
criar_estrutura_diretorios()

# --- Módulo A: consolidação da base ---
# brutos_legitimos = consolidacao.importar_corpus_spamassassin("dados_philder/spamassassin")
# brutos_phishing = consolidacao.importar_corpus_nazario("dados_philder/Nazario.csv")
# df = consolidacao.consolidar_dataset(brutos_legitimos, brutos_phishing)

# --- Módulo B: processamento de texto ---
# df = processamento.pipeline_processamento(df)

# --- Módulo C: balanceamento + treino + teste ---
# df_balanceado = balanceamento.aplicar_undersampling(df)
# df_treino, df_teste = balanceamento.dividir_treino_teste(df_balanceado)
# tokenizer, modelo_tb = balanceamento.carregar_tokenizer_e_modelo()
# dataset_treino, dataset_teste = balanceamento.tokenizar_dataset(df_treino, df_teste, tokenizer)
# trainer, tempo_treino = balanceamento.treinar_modelo(modelo_tb, tokenizer, dataset_treino, dataset_teste)
# predicoes, tempo_teste = balanceamento.testar_modelo(trainer, dataset_teste)
# balanceamento.salvar_modelo(trainer, tokenizer)

# --- Módulo D: avaliação de desempenho ---
# metricas = avaliacao.calcular_metricas_primarias(df_teste["class_label"].tolist(), predicoes)
# eficiencia = avaliacao.calcular_metricas_eficiencia(metricas["accuracy"], tempo_treino, tempo_teste, len(df_teste))
# avaliacao.gerar_curva_perda(trainer.state.log_history)
# avaliacao.salvar_resultados({**metricas, **eficiencia})
```

As linhas comentadas (`#`) mostram exatamente onde cada chamada vai entrar — descomentar conforme cada módulo for preenchido, célula por célula, mantendo o notebook como o "main" que só orquestra, sem lógica própria.

---

## Próximos passos

Aguardando a releitura do artigo para preencher a lógica interna de cada função, incorporando os ajustes de fidelidade ao PHILDER que a Fase 2 pede. A ordem de preenchimento sugerida acompanha a ordem do pipeline e da Parte III do artigo: `database_consolidation.py` (A) → `data_processing.py` (B) → `database_balance.py` (C) → `performance_assessment.py` (D).

Pontos a confirmar na releitura, específicos desta reestruturação:
- Se o artigo detalha algum critério de parsing de `sender`/`subject`/`smtp_headers` além do `body` (o PhishTank/SpamAssassin nem sempre têm cabeçalho SMTP completo e uniforme).
- Se a curva de perda (Módulo D) é de fato reportada no artigo ou é um acréscimo desta replicação para fins didáticos — vale checar antes de apresentar como "fiel ao artigo".
