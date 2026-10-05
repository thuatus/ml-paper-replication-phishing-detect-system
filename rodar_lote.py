"""
rodar_lote.py — execução em lote (sequencial) dos treinamentos da replicação PHILDER.

Fase 3 (04/10/2026). Script independente do notebook: reaproveita os módulos do
pacote `philder` (A–D) exatamente como o notebook os chama, sem alterá-los.

Por que um processo por treinamento
-----------------------------------
O orquestrador chama este mesmo arquivo uma vez por (modelo, rodada), em um
processo Python NOVO. Ao terminar, o processo libera toda a memória — o
equivalente automático a "reiniciar o kernel" entre modelos, prática adotada
no notebook para que um treinamento não herde memória/estado do anterior e os
tempos (TES/IES/RTDE) fiquem comparáveis. Os treinamentos rodam um de cada vez
(nunca em paralelo), para que cada um tenha a máquina inteira.

Uso (a partir da pasta raiz do projeto, onde estão `philder/` e `dados_philder/`)
-------------------------------------------------------------------------------
    python rodar_lote.py --modelos tinybert minilm --rodadas 3
    python rodar_lote.py --modelos todos --rodadas 1
    python rodar_lote.py --modelos todos --rodadas 3 --simular   # só mostra o plano

No Colab/Kaggle (célula):  !python rodar_lote.py --modelos todos --rodadas 1

Saídas (mesmo padrão do notebook)
---------------------------------
    resultados/<modelo>/resultado_final-<rodada>_<ddmmaaaa>.csv
    resultados/<modelo>/curva_perda-<rodada>_<ddmmaaaa>.png
    resultados/<modelo>/log_execucao-<rodada>_<ddmmaaaa>.txt   (saída completa do processo)
    modelo/<modelo>/                                           (modelo final da última rodada)
    resultados/lote_<ddmmaaaa>_<hhmmss>.csv                    (resumo do lote)

O CSV de cada rodada tem as mesmas colunas do notebook, mais colunas de
rastreabilidade: origem, plataforma, processador, gpu, versões de
python/torch/transformers e a impressão digital (MD5) do conteúdo dos conjuntos
balanceado/treino/teste — a mesma verificação que revelou a dependência da ordem
de leitura dos arquivos entre Windows e Linux (Analise_dos_Resultados.md, Seção 8.2).
"""

import argparse
import datetime
import glob
import hashlib
import os
import platform
import subprocess
import sys
import time

# Mesmo dicionário da célula de configuração do notebook (manter sincronizado).
MODELOS_ARTIGO = {
    "tinybert":   "huawei-noah/TinyBERT_General_4L_312D",
    "minilm":     "microsoft/MiniLM-L12-H384-uncased",
    "distilbert": "distilbert-base-uncased",
    "mobilebert": "google/mobilebert-uncased",
    "albert":     "albert-base-v2",
}


def proxima_rodada(pasta: str, data: str) -> int:
    """Próximo número de rodada do dia em `pasta` (1 se ainda não houver nenhuma).
    Mesma lógica da função homônima do notebook."""
    arquivos = glob.glob(os.path.join(pasta, f"resultado_final-*_{data}.csv"))
    numeros = [int(os.path.basename(a).split("-")[1].split("_")[0]) for a in arquivos]
    return max(numeros, default=0) + 1


def _proxima_rodada_lote(pasta: str, data: str) -> int:
    """Como `proxima_rodada`, mas também considera logs de tentativas que falharam
    (que não geram CSV), para nunca sobrescrever o log de uma tentativa anterior."""
    logs = glob.glob(os.path.join(pasta, f"log_execucao-*_{data}.txt"))
    numeros = [int(os.path.basename(a).split("-")[1].split("_")[0]) for a in logs]
    return max(proxima_rodada(pasta, data), max(numeros, default=0) + 1)


def _impressao(df) -> str:
    """MD5 do conteúdo (textos processados, ordenados) — independente da ordem das linhas."""
    texto = "\n".join(sorted(df["body_processado"].astype(str)))
    return hashlib.md5(texto.encode("utf-8")).hexdigest()


def _ambiente() -> dict:
    """Hardware e versões da execução (os tempos só são comparáveis no mesmo hardware)."""
    import torch
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle"):
        plataforma = "kaggle"
    elif os.environ.get("COLAB_RELEASE_TAG") or os.path.isdir("/content/drive"):
        plataforma = "colab"
    else:
        plataforma = f"local-{platform.system().lower()}"
    return {
        "origem": "rodar_lote.py",
        "plataforma": plataforma,
        "processador": platform.processor() or platform.machine(),
        "nucleos": os.cpu_count(),
        "gpu": gpu,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


# ---------------------------------------------------------------------------
# Trabalhador: UM treinamento completo (Módulos A–D), em processo próprio
# ---------------------------------------------------------------------------
def executar_treinamento(modelo_escolhido: str, rodada: int, data_execucao: str) -> None:
    """Reproduz a sequência de chamadas das células A, B, C e D do notebook."""
    from philder import infra
    from philder import database_consolidation as consolidacao
    from philder import data_processing as processamento
    from philder import database_balance as balanceamento
    from philder import performance_assessment as avaliacao

    checkpoint_modelo = MODELOS_ARTIGO[modelo_escolhido]
    pasta_modelo = os.path.join("modelo", modelo_escolhido)
    pasta_checkpoints = os.path.join(pasta_modelo, "checkpoints")
    pasta_resultados = os.path.join("resultados", modelo_escolhido)

    print(f"=== {modelo_escolhido} ({checkpoint_modelo}) — rodada {rodada} — {data_execucao} ===", flush=True)
    infra.criar_estrutura_diretorios()

    # --- Módulo A: consolidação da base ---
    brutos_legitimos = consolidacao.importar_corpus_spamassassin("dados_philder/spamassassin")
    brutos_phishing = consolidacao.importar_corpus_nazario("dados_philder/Nazario.csv")
    legitimos_amostrados, phishing_amostrados = consolidacao.amostrar_quantidade_artigo(
        brutos_legitimos, brutos_phishing, deduplicar_antes=True
    )
    df = consolidacao.consolidar_dataset(legitimos_amostrados, phishing_amostrados)

    # --- Módulo B: processamento de texto ---
    df = processamento.pipeline_processamento(df)

    # --- Módulo C: balanceamento + treino + teste ---
    df_balanceado = balanceamento.aplicar_undersampling(df)
    df_treino, df_teste = balanceamento.dividir_treino_teste(df_balanceado)
    impressoes = {
        "hash_balanceado": _impressao(df_balanceado),
        "hash_treino": _impressao(df_treino),
        "hash_teste": _impressao(df_teste),
    }
    print("Impressões digitais:", impressoes, flush=True)

    tokenizer, modelo_pre_treinado = balanceamento.carregar_tokenizer_e_modelo(checkpoint_modelo)
    dataset_treino, dataset_teste = balanceamento.tokenizar_dataset(df_treino, df_teste, tokenizer)
    trainer, tempo_treino = balanceamento.treinar_modelo(
        modelo_pre_treinado, tokenizer, dataset_treino, dataset_teste, output_dir=pasta_checkpoints
    )
    predicoes, tempo_teste = balanceamento.testar_modelo(trainer, dataset_teste)
    balanceamento.salvar_modelo(trainer, tokenizer, caminho=pasta_modelo)

    # --- Módulo D: avaliação de desempenho ---
    metricas = avaliacao.calcular_metricas_primarias(df_teste["class_label"].tolist(), predicoes)
    tempo_aval = avaliacao.calcular_tempo_avaliacao(trainer.state.log_history)
    eficiencia = avaliacao.calcular_metricas_eficiencia(
        metricas["accuracy"], tempo_treino, tempo_teste, len(df_teste), tempo_avaliacao=tempo_aval
    )
    avaliacao.gerar_curva_perda(
        trainer.state.log_history,
        caminho_imagem=os.path.join(pasta_resultados, f"curva_perda-{rodada}_{data_execucao}.png"),
        titulo=f"Curva de perda — fine-tuning {modelo_escolhido} (rodada {rodada})",
    )
    avaliacao.salvar_resultados(
        {"modelo": modelo_escolhido, "checkpoint": checkpoint_modelo, "rodada": rodada,
         **metricas, **eficiencia, **_ambiente(), **impressoes},
        caminho_csv=os.path.join(pasta_resultados, f"resultado_final-{rodada}_{data_execucao}.csv"),
    )
    print(f"=== Concluído: accuracy={metricas['accuracy']:.4f} | tempo_treino={tempo_treino:.1f}s ===", flush=True)


# ---------------------------------------------------------------------------
# Orquestrador: chama um processo novo por treinamento, em sequência
# ---------------------------------------------------------------------------
def orquestrar(modelos: list, rodadas: int, simular: bool) -> int:
    data_execucao = datetime.date.today().strftime("%d%m%Y")
    plano = [(m, r) for m in modelos for r in range(1, rodadas + 1)]
    print(f"Lote: {len(plano)} treinamento(s) — modelos={modelos}, rodadas por modelo={rodadas}")

    if simular:
        for m in modelos:
            pasta = os.path.join("resultados", m)
            inicio = _proxima_rodada_lote(pasta, data_execucao)
            for i in range(rodadas):
                print(f"  [simulação] {m}: resultados/{m}/resultado_final-{inicio + i}_{data_execucao}.csv")
        return 0

    resumo = []
    os.makedirs("resultados", exist_ok=True)
    caminho_resumo = os.path.join(
        "resultados", f"lote_{data_execucao}_{datetime.datetime.now().strftime('%H%M%S')}.csv"
    )
    proxima_por_modelo = {}  # garante número novo por tentativa, mesmo se uma rodada falhar
    for i, (modelo, _) in enumerate(plano, start=1):
        pasta_resultados = os.path.join("resultados", modelo)
        os.makedirs(pasta_resultados, exist_ok=True)
        rodada = max(_proxima_rodada_lote(pasta_resultados, data_execucao), proxima_por_modelo.get(modelo, 1))
        proxima_por_modelo[modelo] = rodada + 1
        caminho_log = os.path.join(pasta_resultados, f"log_execucao-{rodada}_{data_execucao}.txt")
        print(f"[{i}/{len(plano)}] {modelo} rodada {rodada} — iniciado {datetime.datetime.now():%H:%M:%S} "
              f"(log: {caminho_log})", flush=True)

        ambiente_filho = dict(os.environ, PYTHONIOENCODING="utf-8")
        inicio = time.time()
        with open(caminho_log, "w", encoding="utf-8") as log:
            processo = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--trabalhador",
                 "--modelo", modelo, "--rodada", str(rodada), "--data", data_execucao],
                stdout=log, stderr=subprocess.STDOUT, env=ambiente_filho,
            )
        duracao = time.time() - inicio
        status = "ok" if processo.returncode == 0 else f"ERRO (código {processo.returncode})"
        print(f"         -> {status} em {duracao / 60:.1f} min", flush=True)
        resumo.append({"ordem": i, "modelo": modelo, "rodada": rodada, "status": status,
                       "duracao_processo_s": round(duracao, 1), "log": caminho_log})

        with open(caminho_resumo, "w", encoding="utf-8") as f:  # regrava a cada passo
            f.write("ordem,modelo,rodada,status,duracao_processo_s,log\n")
            for r in resumo:
                f.write(f"{r['ordem']},{r['modelo']},{r['rodada']},{r['status']},"
                        f"{r['duracao_processo_s']},{r['log']}\n")

    falhas = [r for r in resumo if r["status"] != "ok"]
    print(f"Lote concluído: {len(resumo) - len(falhas)} ok, {len(falhas)} com erro. Resumo: {caminho_resumo}")
    return 1 if falhas else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Execução em lote dos treinamentos PHILDER (um processo por treinamento).")
    parser.add_argument("--modelos", nargs="+", default=["tinybert"],
                        help=f"Modelos a treinar ({', '.join(MODELOS_ARTIGO)}) ou 'todos'.")
    parser.add_argument("--rodadas", type=int, default=1, help="Rodadas por modelo (padrão: 1).")
    parser.add_argument("--simular", action="store_true", help="Só mostra o plano, sem treinar.")
    # uso interno (processo trabalhador)
    parser.add_argument("--trabalhador", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--modelo", help=argparse.SUPPRESS)
    parser.add_argument("--rodada", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--data", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not os.path.isdir("philder") or not os.path.isdir("dados_philder"):
        print("Execute a partir da pasta raiz do projeto (onde estão philder/ e dados_philder/).")
        return 2

    if args.trabalhador:
        executar_treinamento(args.modelo, args.rodada, args.data)
        return 0

    modelos = list(MODELOS_ARTIGO) if args.modelos == ["todos"] else args.modelos
    desconhecidos = [m for m in modelos if m not in MODELOS_ARTIGO]
    if desconhecidos:
        print(f"Modelo(s) desconhecido(s): {desconhecidos}. Opções: {list(MODELOS_ARTIGO)} ou 'todos'.")
        return 2
    if args.rodadas < 1:
        print("--rodadas deve ser >= 1.")
        return 2
    return orquestrar(modelos, args.rodadas, args.simular)


if __name__ == "__main__":
    sys.exit(main())
