"""
philder.infra — infraestrutura de apoio do projeto (não é etapa da Parte III).

Diferente dos módulos A/B/C/D (que espelham as etapas da Parte III do
artigo PHILDER), este módulo cuida de infraestrutura de projeto:
criação da estrutura padrão de diretórios usada por todos os outros
módulos (`dados_philder/`, `modelo/`, `modelo/checkpoints/`,
`resultados/`).

Antes vivia solto na raiz do projeto (`preparar_estrutura.py`, Fase 2 —
Passo 1); passou a fazer parte do pacote `philder` para ficar junto dos
demais módulos reutilizáveis (import único, `from philder import infra`),
em vez de depender de um script solto fora do pacote.
"""

import os

# --- Estrutura de diretórios padrão do projeto ---

DATA_DIR = "dados_philder"      # dados brutos e processados (usado desde o Módulo A)
MODELO_DIR = "modelo"           # modelo final e versões intermediárias (checkpoints)
MODELO_CHECKPOINTS_DIR = os.path.join(MODELO_DIR, "checkpoints")
RESULTADOS_DIR = "resultados"   # resultados aferidos (CSV, curva de perda)


def criar_estrutura_diretorios() -> None:
    """Cria a estrutura de diretórios padrão do projeto, se ainda não existir.

    - dados_philder/         -> bases brutas e pré-processadas
    - modelo/                -> modelo final treinado
    - modelo/checkpoints/    -> versões intermediárias do modelo durante o treino
    - resultados/            -> métricas e resultados aferidos, em CSV

    Idempotente: pode ser chamada mais de uma vez sem erro (`exist_ok=True`).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELO_DIR, exist_ok=True)
    os.makedirs(MODELO_CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    print("Estrutura de diretórios criada/confirmada:")
    print(f"  - {DATA_DIR}/")
    print(f"  - {MODELO_DIR}/  (com {MODELO_CHECKPOINTS_DIR}/)")
    print(f"  - {RESULTADOS_DIR}/")


if __name__ == "__main__":
    criar_estrutura_diretorios()
