"""
philder.performance_assessment — Módulo D

Espelha a etapa da Parte III (Metodologia) do artigo PHILDER em que o
desempenho do modelo treinado (Módulo C) é aferido:

    - 4 métricas primárias (preditivas): acurácia, precisão, recall, F1
    - 3 métricas de eficiência: TES, IES, RTDE
    - curva de perda (loss) do treino, para acompanhar a convergência
      do modelo ao longo das épocas

Reaproveita a lógica já validada em
Guia_Pratico_Passo6_Metricas_Basicas.md (Fase 1), formalizando aqui
também o registro da curva de perda, que a versão simplificada da
Fase 1 não gerava.

Dependência nova em relação à Fase 1: `matplotlib` (plot da curva de
perda). Se não estiver instalado no seu ambiente:
    pip install matplotlib
"""

import os

import matplotlib

matplotlib.use("Agg")  # backend sem interface gráfica — só salva em arquivo
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calcular_metricas_primarias(y_true, predicoes) -> dict:
    """Calcula as 4 métricas primárias do artigo comparando `y_true` com
    `predicoes`: acurácia, precisão, recall e F1.

    Retorna um dicionário com essas quatro métricas.
    """
    return {
        "accuracy": accuracy_score(y_true, predicoes),
        "precision": precision_score(y_true, predicoes),
        "recall": recall_score(y_true, predicoes),
        "f1": f1_score(y_true, predicoes),
    }


def calcular_metricas_eficiencia(
    acuracia: float,
    tempo_treino: float,
    tempo_teste: float,
    n_amostras: int,
) -> dict:
    """Calcula as 3 métricas de eficiência do artigo:
    TES (Training Efficiency Score), IES (Inference Efficiency Score) e
    RTDE (Real-Time Detection Efficiency).

    Retorna um dicionário com as três métricas.
    """
    tes = acuracia / tempo_treino #Accuracy / Training Time
    ies = acuracia / tempo_teste  #Accuracy /Testing Time
    rtde = acuracia / (tempo_teste / n_amostras) #Accuracy / Detection Time per Sample
    tempo_deteccao_amostra = (tempo_teste / n_amostras) * 1000
    return {
        "TES": tes,
        "IES": ies,
        "RTDE": rtde,
        "tempo_treino": tempo_treino,
        "tempo_teste": tempo_teste,
        "tempo_deteccao_amostra": tempo_deteccao_amostra,
    }


def gerar_curva_perda(log_history: list, caminho_imagem: str = "resultados/curva_perda.png"):
    """Extrai os valores de `loss` (treino) e `eval_loss` (avaliação por
    época) registrados em `log_history` (`trainer.state.log_history`) e
    plota a curva de perda, salvando a imagem em `caminho_imagem`.

    Retorna o DataFrame com os pontos (step, loss) de treino usados no
    gráfico.
    """
    pontos_treino = [
        {"step": registro["step"], "loss": registro["loss"]}
        for registro in log_history
        if "loss" in registro and "eval_loss" not in registro
    ]
    pontos_avaliacao = [
        {"step": registro["step"], "eval_loss": registro["eval_loss"]}
        for registro in log_history
        if "eval_loss" in registro
    ]

    df_perda_treino = pd.DataFrame(pontos_treino)

    pasta_destino = os.path.dirname(caminho_imagem)
    if pasta_destino:
        os.makedirs(pasta_destino, exist_ok=True)

    plt.figure(figsize=(8, 5))
    if not df_perda_treino.empty:
        plt.plot(df_perda_treino["step"], df_perda_treino["loss"], label="loss (treino)")
    if pontos_avaliacao:
        df_perda_avaliacao = pd.DataFrame(pontos_avaliacao)
        plt.plot(df_perda_avaliacao["step"], df_perda_avaliacao["eval_loss"], label="loss (avaliação)", marker="o")
    plt.xlabel("Passos de treino (step)")
    plt.ylabel("Loss")
    plt.title("Curva de perda — fine-tuning TinyBERT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_imagem)
    plt.close()

    print(f"Curva de perda salva em: {caminho_imagem}")
    return df_perda_treino


def salvar_resultados(resumo: dict, caminho_csv: str = "resultados/resultado_final.csv") -> None:
    """Salva o dicionário de métricas (primárias + eficiência) em
    `caminho_csv`."""
    pasta_destino = os.path.dirname(caminho_csv)
    if pasta_destino:
        os.makedirs(pasta_destino, exist_ok=True)

    pd.DataFrame([resumo]).to_csv(caminho_csv, index=False)
    print(f"Resumo salvo em: {caminho_csv}")
