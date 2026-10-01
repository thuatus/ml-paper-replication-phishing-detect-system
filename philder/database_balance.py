"""
philder.database_balance — Módulo C

Espelha a etapa da Parte III (Metodologia) do artigo PHILDER em que a
base já limpa (Módulo B) é balanceada, tokenizada, dividida em
treino/teste e usada para treinar e testar o modelo:

    1. undersampling: escolha aleatória de mensagens da classe majoritária,
       reduzindo-a até igualar a classe minoritária (aplicado no dataset
       consolidado inteiro, antes do split — assim como no artigo)
    2. tokenização com o tokenizer BERT padrão (o mesmo vocabulário/regras
       usado pelo TinyBERT, por ser destilado do BERT)
    3. split 70/30 (treino/teste)
    4. treino do modelo
    5. teste do modelo (inferência sobre o conjunto de teste)

Reaproveita a lógica já validada em
Guia_Pratico_Passo4_Balanceamento_Classes.md e
Guia_Pratico_Passo5_FineTuning_TinyBERT.md (Fase 1).

`carregar_tokenizer_e_modelo` e `carregar_modelo_treinado` ficam
deliberadamente separadas das funções de treino: são as duas únicas
funções deste módulo que o futuro script standalone de classificação em
Linux e a imagem de inferência enxuta do container (Infra-Container.md)
vão precisar importar — evita duplicar essa lógica em outro lugar.
"""

import time

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)


def aplicar_undersampling(
    df: pd.DataFrame,
    coluna_rotulo: str = "class_label",
    random_state: int = 42,
) -> pd.DataFrame:
    """Reduz a classe majoritária de `df` até igualar a minoritária, por
    escolha randômica das mensagens mantidas (undersampling).

    Retorna um DataFrame balanceado.
    """
    contagem = df[coluna_rotulo].value_counts()
    qtd_minoritaria = contagem.min()

    partes_balanceadas = [
        df[df[coluna_rotulo] == classe].sample(n=qtd_minoritaria, random_state=random_state)
        for classe in contagem.index
    ]

    df_balanceado = pd.concat(partes_balanceadas).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Balanceamento aplicado: {df_balanceado[coluna_rotulo].value_counts().to_dict()}")
    return df_balanceado


def dividir_treino_teste(
    df: pd.DataFrame,
    coluna_rotulo: str = "class_label",
    test_size: float = 0.30,
    random_state: int = 42,
):
    """Divide `df` (já balanceado) em treino/teste na proporção 70/30,
    estratificado por classe.

    Retorna (df_treino, df_teste).
    """
    df_treino, df_teste = train_test_split(
        df,
        test_size=test_size,
        stratify=df[coluna_rotulo],
        random_state=random_state,
    )
    df_treino = df_treino.reset_index(drop=True)
    df_teste = df_teste.reset_index(drop=True)

    print("Treino:", df_treino[coluna_rotulo].value_counts().to_dict())
    print("Teste:", df_teste[coluna_rotulo].value_counts().to_dict())
    return df_treino, df_teste


def carregar_tokenizer_e_modelo(
    modelo_nome: str = "huawei-noah/TinyBERT_General_4L_312D",
    num_labels: int = 2,
):
    """Carrega o tokenizer BERT padrão e o modelo pré-treinado (TinyBERT),
    com a camada de classificação nova (`num_labels` saídas).

    Retorna (tokenizer, modelo).
    """
    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo = AutoModelForSequenceClassification.from_pretrained(modelo_nome, num_labels=num_labels)
    return tokenizer, modelo


def tokenizar_dataset(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    tokenizer,
    coluna_texto: str = "body_processado",
    coluna_rotulo: str = "class_label",
    max_length: int = 128,
):
    """Tokeniza `df_treino`/`df_teste` conforme o padrão do tokenizer
    BERT, produzindo `datasets.Dataset` prontos para o `Trainer`
    (`input_ids`, `attention_mask`, `labels`).

    Retorna (dataset_treino, dataset_teste).
    """
    df_treino = df_treino.dropna(subset=[coluna_texto])
    df_teste = df_teste.dropna(subset=[coluna_texto])

    dataset_treino = Dataset.from_pandas(
        df_treino[[coluna_texto, coluna_rotulo]].rename(columns={coluna_rotulo: "labels"})
    )
    dataset_teste = Dataset.from_pandas(
        df_teste[[coluna_texto, coluna_rotulo]].rename(columns={coluna_rotulo: "labels"})
    )

    def _tokenizar(exemplo):
        return tokenizer(exemplo[coluna_texto], truncation=True, padding="max_length", max_length=max_length)

    dataset_treino = dataset_treino.map(_tokenizar, batched=True)
    dataset_teste = dataset_teste.map(_tokenizar, batched=True)

    return dataset_treino, dataset_teste


def _metricas_durante_treino(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def treinar_modelo(
    modelo,
    tokenizer,
    dataset_treino,
    dataset_teste,
    output_dir: str = "modelo/checkpoints",
    # num_train_epochs: int = 3, --> utilizado como aprendizado inicial 
    num_train_epochs: int = 10, # valor indicado utilizado no artigo
    protocolo: str = "artigo",
):
    """Configura `TrainingArguments`/`Trainer` e executa o fine-tuning do
    modelo sobre `dataset_treino`.

    `protocolo` controla se o conjunto de teste participa de alguma decisão
    de treino (ver discussão do item 1 da Fase 3 — viés de seleção):

    - "artigo" (padrão, Fase 3): fiel ao PHILDER (Figura 6) — treina as
      `num_train_epochs` épocas completas, sem early stopping e sem seleção
      de checkpoint; o modelo final é o da última época. O teste é avaliado
      a cada época **apenas para registrar a curva de perda**
      (`eval_loss`), nenhuma decisão é tomada com base nele. Sem gravação
      de checkpoints por época (`save_strategy="no"`), para não somar I/O
      de disco ao tempo de treino — o modelo final é salvo por
      `salvar_modelo`.
    - "selecao" (Fase 2, lote 25/09(b)): early stopping (patience 2) e
      recarga do checkpoint de melhor F1 — ambos decididos sobre o
      conjunto de teste. Mantido só para reproduzir o resultado anterior e
      medir o viés de seleção por comparação com o protocolo "artigo".

    Hiperparâmetros não especificados pelo artigo (learning rate, batch
    size, weight decay) seguem os valores já usados na Fase 2:
    lr padrão do `Trainer` (5e-5), batch 8, weight_decay 0.01.

    Retorna (trainer, tempo_treino_segundos). O tempo é medido em volta de
    `trainer.train()` inteiro (inclui a avaliação por época); o tempo gasto
    só na avaliação pode ser extraído depois com
    `performance_assessment.calcular_tempo_avaliacao(trainer.state.log_history)`.
    """
    if protocolo not in ("artigo", "selecao"):
        raise ValueError(f"protocolo deve ser 'artigo' ou 'selecao', recebido: {protocolo!r}")

    args_comuns = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        weight_decay=0.01,
        logging_steps=10,
        dataloader_pin_memory=False,
        report_to="none",
    )

    if protocolo == "artigo":
        args_treino = TrainingArguments(**args_comuns, save_strategy="no")
        callbacks = []
    else:
        ## Ajuste de configurações para mitigação de overfiting do treinamento (Fase 2)
        args_treino = TrainingArguments(
            **args_comuns,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer = Trainer(
        model=modelo,
        args=args_treino,
        train_dataset=dataset_treino,
        eval_dataset=dataset_teste,
        compute_metrics=_metricas_durante_treino,
        callbacks=callbacks,
    )

    inicio = time.time()
    trainer.train()
    tempo_treino = time.time() - inicio
    print(
        f"Tempo de treino: {tempo_treino:.1f}s "
        f"[protocolo={protocolo}, épocas executadas: {trainer.state.epoch:.0f}]"
    )

    return trainer, tempo_treino


def testar_modelo(trainer, dataset_teste):
    """Roda a inferência do modelo treinado sobre `dataset_teste`
    (etapa de teste do modelo, distinta da avaliação por época feita
    durante o treino).

    Retorna (predicoes, tempo_teste_segundos).
    """
    inicio = time.time()
    saida = trainer.predict(dataset_teste)
    tempo_teste = time.time() - inicio

    predicoes = np.argmax(saida.predictions, axis=-1).tolist()
    print(f"Tempo total de inferência no teste: {tempo_teste:.2f}s")
    return predicoes, tempo_teste


def salvar_modelo(trainer, tokenizer, caminho: str = "modelo") -> None:
    """Salva o modelo treinado e o tokenizer em `caminho` (config, pesos,
    vocabulário)."""
    trainer.save_model(caminho)
    tokenizer.save_pretrained(caminho)
    print(f"Modelo salvo em: {caminho}/")


def carregar_modelo_treinado(caminho: str = "modelo"):
    """Recarrega um modelo e tokenizer já treinados e salvos em `caminho`,
    prontos para inferência (`.eval()`).

    Usado pelo Módulo D (avaliação), pelo futuro script standalone de
    classificação em Linux e pela imagem de inferência do container.

    Retorna (tokenizer, modelo).
    """
    tokenizer = AutoTokenizer.from_pretrained(caminho)
    modelo = AutoModelForSequenceClassification.from_pretrained(caminho)
    modelo.eval()
    return tokenizer, modelo
