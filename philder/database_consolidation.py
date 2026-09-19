"""
philder.database_consolidation — Módulo A

Espelha a etapa da Parte III (Metodologia) do artigo PHILDER em que as
fontes de dados brutas são importadas e consolidadas em uma única base,
já com o rótulo de classificação atribuído.

Fontes utilizadas nesta replicação:
    - SpamAssassin Public Corpus  -> e-mails legítimos (ham)
    - Nazario Phishing Corpus     -> e-mails de phishing
      (substitui o PhishTank do artigo original, indisponível para coleta
      em lote no momento da replicação — decisão já registrada nos guias
      da Fase 1, Passo 2)

Saída consolidada (schema-alvo, um registro por e-mail):
    sender          — remetente (campo "From" do cabeçalho SMTP)
    subject         — assunto (campo "Subject" do cabeçalho SMTP)
    body            — corpo da mensagem (texto), sem o cabeçalho
    smtp_headers    — cabeçalho SMTP bruto, preservado à parte
    class_label     — rótulo binário (0 = legítimo, 1 = phishing)
"""

import email
import glob
import os
import re
import tarfile
import urllib.request

import pandas as pd

# Fonte oficial do corpus (mesma usada na Fase 1 — Guia_Pratico_Passo2_Obtencao_Dados.md)
_DOWNLOAD_ROOT = "https://spamassassin.apache.org/old/publiccorpus/"
_HAM_URL = _DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
_PASTA_HAM_PADRAO = "easy_ham"  # nome da subpasta criada ao extrair o .tar.bz2

# Nomes de coluna mais comuns encontrados nas diferentes versões do
# Nazario Phishing Corpus em CSV (ex.: dataset "Phishing Email Dataset" do Kaggle)
_COLUNAS_TEXTO_CANDIDATAS = ["text", "body", "Email Text", "email_text"]


def importar_corpus_spamassassin(caminho_pasta: str) -> list[str]:
    """Baixa (se necessário), extrai e lê os e-mails legítimos ("ham") do
    SpamAssassin Public Corpus a partir/dentro de `caminho_pasta`.

    Idempotente: se o `.tar.bz2` já foi baixado, não baixa de novo; se a
    pasta já foi extraída, não extrai de novo — só relê os arquivos.

    Cada e-mail é mantido como mensagem bruta (texto completo, cabeçalho +
    corpo), para ser decomposto depois por `extrair_campos_email`. Usa
    `encoding="latin-1"` na leitura porque e-mails desse corpus (anos 2000)
    frequentemente têm bytes fora do padrão UTF-8.

    Retorna uma lista de strings (uma por e-mail bruto).
    """
    os.makedirs(caminho_pasta, exist_ok=True)

    nome_arquivo = _HAM_URL.split("/")[-1]
    caminho_compactado = os.path.join(caminho_pasta, nome_arquivo)
    if not os.path.isfile(caminho_compactado):
        urllib.request.urlretrieve(_HAM_URL, caminho_compactado)
        print(f"Baixado: {nome_arquivo}")

    pasta_extraida = os.path.join(caminho_pasta, _PASTA_HAM_PADRAO)
    if not os.path.isdir(pasta_extraida):
        with tarfile.open(caminho_compactado, "r:bz2") as tar:
            tar.extractall(path=caminho_pasta)
        print(f"Extraído em: {caminho_pasta}")

    arquivos_ham = glob.glob(os.path.join(pasta_extraida, "*"))
    emails_brutos = []
    for caminho in arquivos_ham:
        if os.path.isfile(caminho):
            with open(caminho, "r", encoding="latin-1") as f:
                emails_brutos.append(f.read())

    print(f"Total de e-mails legítimos carregados: {len(emails_brutos)}")
    return emails_brutos


def importar_corpus_nazario(caminho_csv: str, coluna_texto: str = "text") -> list[str]:
    """Lê os e-mails de phishing do Nazario Phishing Corpus a partir do
    CSV em `caminho_csv` (substituto do PhishTank nesta replicação).

    Se `coluna_texto` não existir no CSV, tenta identificar automaticamente
    a coluna certa entre os nomes mais comuns encontrados nas diferentes
    versões públicas do corpus (`text`, `body`, `Email Text`, ...).

    Retorna uma lista de strings (uma por e-mail bruto), no mesmo formato
    produzido por `importar_corpus_spamassassin`, para permitir o mesmo
    parsing em `extrair_campos_email`.
    """
    df_bruto = pd.read_csv(caminho_csv)

    if coluna_texto not in df_bruto.columns:
        coluna_encontrada = next(
            (c for c in _COLUNAS_TEXTO_CANDIDATAS if c in df_bruto.columns), None
        )
        if coluna_encontrada is None:
            raise ValueError(
                f"Coluna de texto '{coluna_texto}' não encontrada em '{caminho_csv}'. "
                f"Colunas disponíveis: {df_bruto.columns.tolist()}. "
                "Informe o nome correto pelo parâmetro coluna_texto."
            )
        coluna_texto = coluna_encontrada

    emails_brutos = df_bruto[coluna_texto].dropna().astype(str).tolist()
    print(f"Total de e-mails de phishing carregados: {len(emails_brutos)} (coluna: '{coluna_texto}')")
    return emails_brutos


def extrair_campos_email(mensagem_bruta: str) -> dict:
    """Decompõe uma mensagem de e-mail bruta (cabeçalho + corpo) nos
    campos do schema-alvo: `sender`, `subject`, `body`, `smtp_headers`.

    Usa o módulo padrão `email` para extrair `From`/`Subject` do
    cabeçalho SMTP, e a mesma lógica de separação cabeçalho/corpo já
    validada na Fase 1 (Passo 3): tudo antes da primeira linha em branco
    é cabeçalho, o restante é corpo. Mensagens sem cabeçalho reconhecível
    (comum em CSVs já processados, como o Nazario) caem no caso de
    fallback: `sender`/`subject`/`smtp_headers` ficam vazios e o texto
    inteiro é tratado como `body`.

    Retorna um dicionário com essas quatro chaves.
    """
    msg = email.message_from_string(mensagem_bruta)
    sender = (msg.get("From") or "").strip()
    subject = (msg.get("Subject") or "").strip()

    partes = re.split(r"\n\s*\n", mensagem_bruta, maxsplit=1)
    if len(partes) == 2:
        smtp_headers, body = partes
    else:
        smtp_headers, body = "", mensagem_bruta

    return {
        "sender": sender,
        "subject": subject,
        "body": body.strip(),
        "smtp_headers": smtp_headers.strip(),
    }


def criar_rotulo_classificacao(campos_email: dict, eh_phishing: bool) -> dict:
    """Acrescenta o campo `class_label` (0 = legítimo, 1 = phishing) ao
    dicionário `campos_email` já extraído por `extrair_campos_email`.

    Retorna o dicionário de entrada com a chave `class_label` adicionada.
    """
    campos_email = dict(campos_email)
    campos_email["class_label"] = 1 if eh_phishing else 0
    return campos_email


def consolidar_dataset(
    emails_legitimos_brutos: list[str],
    emails_phishing_brutos: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Aplica `extrair_campos_email` + `criar_rotulo_classificacao` a
    todos os e-mails brutos das duas fontes e consolida o resultado em
    um único DataFrame, embaralhado.

    Retorna um DataFrame com as colunas:
    `sender, subject, body, smtp_headers, class_label`.
    """
    registros = []
    for mensagem_bruta in emails_legitimos_brutos:
        campos = extrair_campos_email(mensagem_bruta)
        registros.append(criar_rotulo_classificacao(campos, eh_phishing=False))
    for mensagem_bruta in emails_phishing_brutos:
        campos = extrair_campos_email(mensagem_bruta)
        registros.append(criar_rotulo_classificacao(campos, eh_phishing=True))

    df = pd.DataFrame(registros, columns=["sender", "subject", "body", "smtp_headers", "class_label"])
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    contagem = df["class_label"].value_counts().to_dict()
    print(f"Dataset consolidado: {len(df)} registros — legítimos: {contagem.get(0, 0)}, phishing: {contagem.get(1, 0)}")
    return df


def carregar_dataset(caminho_csv: str) -> pd.DataFrame:
    """Recarrega um dataset já consolidado (schema-alvo) salvo em disco,
    sem precisar reprocessar as fontes brutas novamente.
    """
    df = pd.read_csv(caminho_csv)
    print(f"Dataset carregado: {df.shape[0]} registros, colunas: {df.columns.tolist()}")
    return df
