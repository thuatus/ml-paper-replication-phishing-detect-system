"""
philder — módulos reutilizáveis da replicação do PHILDER (Fase 2).

Estrutura reorganizada para espelhar os "capítulos" da Parte III
(Metodologia) do artigo PHILDER, um módulo por etapa:

    A -> database_consolidation  -> import das fontes + rótulo de classificação
                                     (sender, subject, body, smtp_headers, class_label)
    B -> data_processing         -> remoção de cabeçalho SMTP, substituição de
                                     URLs, normalização de texto, stopwords
    C -> database_balance        -> undersampling, tokenização (BERT padrão),
                                     split 70/30, treino e teste do modelo
    D -> performance_assessment  -> 4 métricas primárias, 3 métricas de
                                     eficiência (TES/IES/RTDE), curva de perda

Além das etapas da Parte III, o pacote também inclui:

    infra -> infraestrutura de apoio (fora da Parte III): criação da
             estrutura padrão de diretórios do projeto
             (dados_philder/, modelo/, modelo/checkpoints/, resultados/)

Importado tanto pelo notebook de treino/experimentação quanto, futuramente,
pelo script standalone de classificação em Linux e pela imagem de inferência
do container (ver Infra-Container.md). Para esse reaproveitamento,
`database_balance.carregar_tokenizer_e_modelo` e
`database_balance.carregar_modelo_treinado` ficam separadas do restante do
Módulo C — são as únicas funções que a imagem de inferência enxuta precisa
importar, evitando duplicar/dessincronizar essa lógica em outro lugar.

Todas as funções de A/B/C/D e de `infra` estão implementadas (não são
mais esqueleto/`NotImplementedError`), reaproveitando a lógica já
validada na Fase 1 (Passos 1-6), adaptada ao schema
`sender, subject, body, smtp_headers, class_label` e à ordem de etapas
do artigo. Única dependência nova em relação à Fase 1: `matplotlib`
(curva de perda, usado em `performance_assessment.gerar_curva_perda`)
— instale com `pip install matplotlib` no seu ambiente (`env_philder1`)
se ainda não tiver. A remoção de stopwords (`data_processing.
remover_stopwords`) usa uma lista de stopwords em inglês embutida como
constante (a mesma lista padrão do NLTK), em vez de depender de `nltk`
+ download em tempo de execução — evita mais um ponto de fricção de
rede/proxy no Windows, como o que já aconteceu ao instalar pacotes na
Fase 1.
"""
