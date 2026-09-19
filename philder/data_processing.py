"""
philder.data_processing — Módulo B

Espelha a etapa da Parte III (Metodologia) do artigo PHILDER em que o
texto de cada e-mail (já consolidado pelo Módulo A) é limpo e
normalizado antes da tokenização/treino:

    1. remoção do cabeçalho SMTP (defesa extra — o Módulo A já separa
       `body` de `smtp_headers` na extração)
    2. substituição de URLs por um placeholder genérico
    3. normalização do texto (minúsculas + remoção de caracteres especiais)
    4. remoção de stopwords

Reaproveita a lógica de limpeza já validada em
Guia_Pratico_Passo3_Preprocessamento.md (Fase 1), acrescentando aqui a
remoção de stopwords que a versão simplificada da Fase 1 havia deixado
de fora.

Sem dependência nova em relação à Fase 1: a lista de stopwords em inglês
usada por `remover_stopwords` é a lista padrão do NLTK (`stopwords.words
("english")`) embutida como constante (`STOPWORDS_INGLES`) — evita
depender de `nltk` + `nltk.download(...)` em tempo de execução (rede
+ proxy no Windows já deu trabalho neste projeto durante a instalação
de pacotes na Fase 1).
"""

import re

import pandas as pd

_REGEX_URL = re.compile(r"https?://\S+|www\.\S+")
_REGEX_LINHA_CABECALHO = re.compile(r"^[A-Za-z][A-Za-z\-]*:\s?.*$")

# Lista padrão de stopwords em inglês do NLTK (nltk.corpus.stopwords.words
# ("english")), copiada como constante para não depender de download em
# tempo de execução. As contrações da lista original (ex.: "don't",
# "you're") já vêm aqui separadas em suas partes ("don", "t", "you",
# "re") — é exatamente o formato que sobra depois que `normalizar_texto`
# troca o apóstrofo por espaço (ex.: "don't" -> "don t"), então os
# tokens curtos (s, t, d, ll, m, o, re, ve, y) cobrem essas partes.
STOPWORDS_INGLES = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself
    yourselves he him his himself she her hers herself it its itself
    they them their theirs themselves what which who whom this that
    these those am is are was were be been being have has had having do
    does did doing a an the and but if or because as until while of at
    by for with about against between into through during before after
    above below to from up down in out on off over under again further
    then once here there when where why how all any both each few more
    most other some such no nor not only own same so than too very s t
    can will just don should now d ll m o re ve y ain aren couldn didn
    doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn
    weren won wouldn
    """.split()
)


def remover_cabecalho_smtp(body: str, smtp_headers: str) -> str:
    """Garante que nenhum resquício do cabeçalho SMTP permaneça em `body`
    (defesa extra: o Módulo A já separa `body` de `smtp_headers` na
    extração, mas cabeçalhos aninhados/citados em replies podem persistir
    no corpo do e-mail).

    Remove, do início de `body`, qualquer sequência de linhas no formato
    "Chave: valor" (padrão de cabeçalho SMTP) — típico de e-mails
    encaminhados/citados que trazem um segundo bloco de cabeçalho dentro
    do corpo.

    Retorna o `body` limpo de cabeçalho.
    """
    if not body:
        return body

    linhas = body.split("\n")
    i = 0
    while i < len(linhas) and _REGEX_LINHA_CABECALHO.match(linhas[i].strip()):
        i += 1

    # só remove se encontrou pelo menos uma linha de cabeçalho residual
    corpo_limpo = "\n".join(linhas[i:]) if i > 0 else body
    return corpo_limpo.strip()


def substituir_urls(texto: str, placeholder: str = "URL") -> str:
    """Localiza URLs no texto (regex) e as substitui por `placeholder`,
    evitando que o modelo aprenda a decorar domínios específicos em vez
    do padrão textual do phishing.

    Retorna o texto com as URLs substituídas.
    """
    return _REGEX_URL.sub(f" {placeholder} ", texto)


def normalizar_texto(texto: str) -> str:
    """Normaliza o texto: converte para minúsculas e remove caracteres
    especiais/pontuação, colapsando espaços em excesso.

    Retorna o texto normalizado.
    """
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def remover_stopwords(texto: str) -> str:
    """Remove stopwords em inglês do `texto`, usando a lista padrão do
    NLTK embutida em `STOPWORDS_INGLES` (o corpus do PHILDER é
    majoritariamente em inglês).

    Espera texto já normalizado (minúsculas, sem pontuação) — rode depois
    de `normalizar_texto` no pipeline.

    Retorna o texto sem as stopwords.
    """
    tokens = texto.split()
    tokens_filtrados = [t for t in tokens if t not in STOPWORDS_INGLES]
    return " ".join(tokens_filtrados)


def pipeline_processamento(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica, em sequência, `remover_cabecalho_smtp` -> `substituir_urls`
    -> `normalizar_texto` -> `remover_stopwords` a cada linha de `df`
    (colunas `body`, `smtp_headers` produzidas pelo Módulo A).

    Retorna o DataFrame de entrada com uma nova coluna `body_processado`.
    """
    df = df.copy()

    def _processar_linha(linha) -> str:
        texto = remover_cabecalho_smtp(linha["body"], linha["smtp_headers"])
        texto = substituir_urls(texto)
        texto = normalizar_texto(texto)
        texto = remover_stopwords(texto)
        return texto

    df["body_processado"] = df.apply(_processar_linha, axis=1)

    vazios = (df["body_processado"].str.strip() == "").sum()
    print(f"Pré-processamento concluído: {len(df)} registros ({vazios} ficaram vazios)")
    return df
