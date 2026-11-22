# PHILDER Replication — Phishing Detection with Compact Models

Independent replication of the paper **[PHILDER: Lightweight Framework for Intelligent Phishing Detection on Resource-Limited Devices](https://ieeexplore.ieee.org/abstract/document/11095665)** (BISPO et al., 2025), developed as a project for the Distributed Applications course of the Graduate Program in Electrical Engineering (PPGEE) at the University of Brasília (UnB), Brazil.

PHILDER classifies the body of e-mails as *phishing* or legitimate by combining the compact language model **TinyBERT** with class balancing through ***undersampling***. This repository reproduces the *pipeline* described in the paper's methodology, compares the five models evaluated by the authors, and repeats the ablation study, with batch execution and data traceability.

> The code, file names, folder names and command-line options are in Portuguese. Their meaning is given in the tables below.

---

## Table of contents

- [Main results](#main-results)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Data](#data)
- [How to run](#how-to-run)
- [Reproducibility](#reproducibility)
- [Paper](#paper)
- [How to cite](#how-to-cite)
- [Author and context](#author-and-context)
- [License](#license)

---

## Main results

Official batch of Nov 1–2, 2026: 5 models × 6 runs, same test set (301 e-mails: 151 legitimate and 150 *phishing*), CPU only.

| Model | Accuracy — PHILDER | Accuracy — this work | F1 — this work | Training (s) | Detection (ms/e-mail) |
| --- | --- | --- | --- | --- | --- |
| **TinyBERT (PHILDER)** | 0.9927 ± 0.0006 | **0.9939 ± 0.0039** | **0.9939 ± 0.0038** | **392.3** | **7.31** |
| DistilBERT | 0.9934 ± 0.0005 | 0.9906 ± 0.0044 | 0.9906 ± 0.0043 | 1,699.6 | 53.99 |
| MobileBERT | 0.9904 ± 0.0004 | 0.9906 ± 0.0033 | 0.9906 ± 0.0032 | 1,466.6 | 36.09 |
| MiniLM | 0.9889 ± 0.0007 | 0.9884 ± 0.0028 | 0.9884 ± 0.0027 | 1,430.8 | 30.23 |
| ALBERT | 0.9834 ± 0.0008 | 0.9878 ± 0.0054 | 0.9879 ± 0.0053 | 3,085.0 | 124.25 |

- The classification performance reported in the paper was reproduced: all five models were within 1.3 e-mails of the reported means.
- TinyBERT achieved the highest accuracy and was the most efficient model: training 3.6 to 7.9 times faster and detection 4 to 17 times faster than the other models, on the same *hardware*.
- The ablation study confirmed the contribution of pre-training and *undersampling*, with smaller effects than those reported in the paper.
- Internal inconsistencies were found in the efficiency metrics (TES, IES and RTDE) published in the original paper.

Methods, results and discussion are described in the paper (in Portuguese): [`Artigo_Replicacao_PHILDER.pdf`](Artigo_Replicacao_PHILDER.pdf).

---

## Repository structure

```
.
├── philder/                       # Python package: one module per methodology step (Part III of the paper)
│   ├── infra.py                   #   Directory structure
│   ├── database_consolidation.py  #   Module A — dataset import, deduplication, sampling and labeling
│   ├── data_processing.py         #   Module B — SMTP header, URL/EMAILADDRESS masks, normalization, stopwords
│   ├── database_balance.py        #   Module C — undersampling, 70/30 split, tokenization, training and testing
│   └── performance_assessment.py  #   Module D — classification and efficiency metrics, loss curve
├── philder.ipynb                  # Main notebook: orchestrates the modules, cell by cell
├── rodar_lote.py                  # Batch execution: one process per training run
├── requirements.txt               # Dependencies
├── LICENSE.md                     # License (PolyForm Noncommercial 1.0.0)
├── resultados/                    # Results per model (CSV files and loss curves), batches and ablation
│   ├── <model>/                   #   tinybert, distilbert, mobilebert, minilm, albert
│   ├── ablacao/                   #   Ablation: sem_pretreino (no pre-training), sem_undersampling (no undersampling)
│   └── lote_*.csv                 #   Summary of each batch execution
├── dados_philder/                 # E-mail datasets (not versioned; see "Data")
├── modelo/                        # Trained models (not versioned)
└── Artigo_Replicacao_PHILDER.pdf  # Paper describing the replication (final version, in Portuguese)
```

---

## Requirements

| Component | Version used |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.14 (CPU) |
| `transformers` (Hugging Face) | 5.16 |
| Other libraries | `accelerate`, `datasets`, `scikit-learn`, `pandas`, `numpy`, `matplotlib` |
| Development environment | JupyterLab |

The official results were obtained on a Windows laptop with an Intel i7-8750H (6 cores, 12 *threads*) and 16 GB of memory, without GPU.

Suggested installation in a virtual environment:

```bash
python -m venv venv_philder
venv_philder\Scripts\activate          # Windows
# source venv_philder/bin/activate     # Linux/macOS
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

*Stopword* removal uses the standard NLTK English list embedded in the code, without depending on the `nltk` library.

---

## Data

The datasets are not versioned (see `.gitignore`). The SpamAssassin file is downloaded automatically from the [project's public repository](https://spamassassin.apache.org/old/publiccorpus/) on the first run; `Nazario.csv` must be placed manually in `dados_philder/`:

```
dados_philder/
├── Nazario.csv                          # Phishing e-mails (Nazario corpus, Kaggle CSV version)
└── spamassassin/
    └── 20030228_easy_ham.tar.bz2        # Legitimate e-mails (SpamAssassin Public Corpus, easy_ham)
```

| Dataset | Source | Raw | Duplicates removed | Unique |
| --- | --- | --- | --- | --- |
| *Phishing* | Nazario corpus | 1,565 | 94 | 1,471 |
| Legitimate | SpamAssassin `easy_ham` | 2,501 | 55 | 2,446 |

PhishTank, the source cited in the paper, only provides URLs, not the text of the e-mails; it was therefore replaced by the Nazario corpus. Deduplication is performed on the pre-processed text, before sampling. After *undersampling*, the dataset has 502 legitimate and 501 *phishing* e-mails, split into 702 for training and 301 for testing.

---

## How to run

### From the notebook

Open `philder.ipynb` in JupyterLab and run the cells in order. The configuration (model, *undersampling*) is set in the first cells.

### In batch

`rodar_lote.py` runs each training in a separate process, sequentially, and numbers the runs automatically. Without additional options, it reproduces the official configuration.

```bash
# Official batch: 5 models × 6 runs
python rodar_lote.py --modelos todos --rodadas 6

# TinyBERT only, one run
python rodar_lote.py --modelos tinybert --rodadas 1

# Show the execution plan without training
python rodar_lote.py --modelos todos --rodadas 6 --simular

# Ablation study (TinyBERT)
python rodar_lote.py --modelos tinybert --rodadas 6 --ablacao sem_pretreino
python rodar_lote.py --modelos tinybert --rodadas 6 --ablacao sem_undersampling
```

| Option | Values | Effect |
| --- | --- | --- |
| `--modelos` (models) | `tinybert`, `distilbert`, `mobilebert`, `minilm`, `albert` or `todos` (all) | Models to train |
| `--rodadas` (runs) | integer | Runs per model |
| `--undersampling` | `artigo` (paper, 502/501, default), `minoritaria` (minority class size, 501/501) or `L,P` (legitimate, phishing) | Size of each class |
| `--ablacao` (ablation) | `sem_pretreino` (no pre-training) or `sem_undersampling` (no undersampling) | Removes one component of PHILDER; outputs in `resultados/ablacao/<type>/` |
| `--simular` (simulate) | — | Shows the plan without training |

Checkpoints used (Hugging Face Hub):

| Model | Checkpoint |
| --- | --- |
| TinyBERT | `huawei-noah/TinyBERT_General_4L_312D` |
| DistilBERT | `distilbert-base-uncased` |
| MobileBERT | `google/mobilebert-uncased` |
| MiniLM | `microsoft/MiniLM-L12-H384-uncased` |
| ALBERT | `albert-base-v2` |

---

## Reproducibility

| Item | Configuration |
| --- | --- |
| Training protocol | 10 fixed epochs, no early stopping and no *checkpoint* selection; the test set is used only at the end |
| Hyperparameters (same for all models) | Learning rate 5e-5, batch size 8, `weight_decay` 0.01, `max_length` 128, no *warmup* |
| Split | Stratified 70/30 |
| Seeds | 42 in every random step (sampling, split and training) |
| File reading order | Fixed with `sorted()` (Windows and Linux return files in different orders) |
| Isolation | One Python process per training run, sequential execution |
| Traceability | Each result CSV records the environment, library versions and MD5 hashes of the datasets |

Reference MD5 fingerprints (official batch):

| Set | Size | MD5 |
| --- | --- | --- |
| Balanced | 1,003 (502 / 501) | `ba57e8a386cd073e289814fc3f932b0f` |
| Training | 702 (351 / 351) | `ab6aef267000c247dae0b1ece4ccae85` |
| Test | 301 (151 / 150) | `ccfa3c3612628cdfcee16eab995a7d00` |

If the MD5 hashes you obtain are different, the data or the pre-processing are not the same, and the results are not directly comparable.

---

## Paper

The replication is described in the paper below (final version, in Portuguese, with an abstract in English):

> **Detecção de Phishing em Dispositivos com Recursos Limitados: um Estudo de Reprodutibilidade com Modelos Compactos** (*Phishing Detection on Resource-Limited Devices: a Reproducibility Study with Compact Models*). Alvaro Santana dos Santos Júnior and Vinícius Pereira Gonçalves. University of Brasília (UnB), 2026.
>
> PDF: [`Artigo_Replicacao_PHILDER.pdf`](Artigo_Replicacao_PHILDER.pdf)

---

## How to cite

This work:

> SANTOS JÚNIOR, A. S. dos; GONÇALVES, V. P. **Detecção de Phishing em Dispositivos com Recursos Limitados**: um Estudo de Reprodutibilidade com Modelos Compactos. Brasília: Universidade de Brasília, 2026.

Original paper:

> BISPO, G. D. et al. PHILDER: lightweight framework for intelligent phishing detection on resource-limited devices. **IEEE Access**, v. 13, p. 131967-131979, 2025.

---

## Author and context

Developed by **Alvaro Santana dos Santos Júnior** as a hands-on scientific paper replication project for the **Distributed Applications** course — PPGEE/UnB, 2026.

## License

This project is distributed under the **[PolyForm Noncommercial License 1.0.0](LICENSE.md)**:

- **Allowed free of charge:** use, copying, modification and redistribution for noncommercial purposes, such as research, teaching, personal study, and use by educational and research institutions and nonprofit organizations.
- **Commercial use:** not granted by this license. To use the software in a commercial product, service or activity, contact the author to negotiate a commercial license.

The dependencies (PyTorch, Hugging Face `transformers`, etc.), the pre-trained models and the datasets (Nazario corpus and SpamAssassin) are subject to their own licenses and terms of use.
