# Ideias para a Dissertação de Mestrado — Handoff do chat da replicação PHILDER

> **Para o chat "Ideias dissertação Mestrado":** este documento é autocontido. Ele traz uma ideia nova, nascida na replicação do PHILDER, com o contexto necessário para avaliá-la e consolidá-la no compilado `ideias-pesquisa-mestrado.md`, junto das ideias já registradas lá. A Seção 4 sugere como as ideias se conectam.
>
> Versão 2 — 10/11/2026. Fonte dos números: `Analise_dos_Resultados.md` (lote oficial de 01–02/11/2026 e ablação de 06–07/11/2026), Project "Aplicações DIstribuídas".

---

## 1. Contexto mínimo (o que já foi feito)

- **Artigo-base:** Bispo et al., *PHILDER: Lightweight Framework for Intelligent Phishing Detection on Resource-Limited Devices*, IEEE Access, v. 13, 2025. PHILDER = TinyBERT + undersampling para classificar e-mails como phishing ou legítimos.
- **Replicação concluída** na disciplina Aplicações Distribuídas (PPGEE/UnB):
  - dados Nazario (phishing) + Enron (legítimos), com undersampling 502/501 e teste de 301 e-mails;
  - 5 modelos, 6 rodadas cada, 10 épocas, num notebook Dell G7 (i7-8750H, GTX 1050 Ti).
- **Resultado oficial (TinyBERT):**
  - acurácia 0,9939 ± 0,0039 (artigo: 0,9927);
  - tempo de detecção ≈ 7,3 ms por e-mail;
  - os 5 modelos ficaram a menos de 1,3 e-mail de diferença do artigo.
- **Ablação (TinyBERT):**
  - sem pré-treino: acurácia 0,9801, com os falsos negativos subindo de 0,3 para 5,3 por rodada;
  - sem undersampling: acurácia balanceada 0,9954, mas o treino ficou cerca de 3× mais lento.
- **Problemas encontrados no artigo original:**
  - desvios-padrão impossíveis para um teste de 301 amostras (±0,0006);
  - a métrica TES aparece em três escalas diferentes;
  - Tabela 5 e Figura 3 se contradizem;
  - acurácia e F1 incompatíveis na linha sem undersampling.

  Conclusão: **as métricas de eficiência do artigo não são confiáveis como estão publicadas.**

---

## 2. A lacuna que motiva a ideia

O artigo afirma que o PHILDER é adequado a **"dispositivos com recursos limitados"** e usa métricas compostas para sustentar isso:

- TES = acurácia ÷ tempo de treino;
- IES = acurácia ÷ tempo de teste;
- RTDE = acurácia ÷ tempo de detecção.

Porém, **todas as medições foram feitas em PC/GPU**: o modelo nunca foi executado num dispositivo limitado. A replicação tem a mesma limitação. Os 7,3 ms por e-mail num i7 não dizem nada sobre um Raspberry Pi, um roteador ou um firewall pequeno.

---

## 3. Ideia — Implantação e medição real do PHILDER na borda

### 3.1 Pergunta central

**O detector treinado, empacotado para a borda (ONNX/TFLite, quantizado), mantém a acurácia e atende ao volume real de e-mails em hardware de baixo custo? E em que ponto da rede ele faz sentido?**

### 3.2 Análise preliminar de viabilidade (estimativas, não medições)

**Requisitos do modelo só para detecção (TinyBERT 4L-312D):**
- pesos de ≈ 57 MB em fp32, ≈ 29 MB em fp16 e ≈ 15 MB em int8;
- ≈ 1,2 GFLOPs por sequência de 128 tokens;
- o runtime PyTorch ocupa centenas de MB, enquanto ONNX Runtime ou TFLite ocupam de poucos MB a dezenas de MB.

| Equipamento | Arquitetura típica | Viável só para detecção? | Observação |
|---|---|---|---|
| Microcontroladores (ESP32, Cortex-M) | RISC-V / ARM Cortex-M | **Não** | KB a poucos MB de memória; o modelo não cabe |
| Microcontroladores com NPU (Ethos-U) | ARM Cortex-M + NPU | **Não** | NPU feita para modelos de poucos MB |
| Switches gerenciáveis | MIPS / ARM / ASIC | **Não** | Não veem o e-mail montado, só pacotes |
| Roteadores domésticos (128–256 MB) | MIPS | **No limite** | Só int8 com runtime leve; ≈ segundos por e-mail |
| Roteadores ARM (512 MB–1 GB) | ARM Cortex-A53 | **Sim, tecnicamente** | ≈ 0,1–1 s; raramente veem o conteúdo do e-mail |
| Raspberry Pi e gateways ARM | ARM Cortex-A72/A76 | **Sim** | ≈ dezenas a centenas de ms |
| Firewalls pequenos | ARM ou x86 | **Sim** | Lugar natural, se fizerem inspeção de e-mail |
| Gateways x86 | x86-64 | **Sim** | ≈ 10–30 ms |
| Jetson | ARM + GPU | **Sim, com folga** | Poucos ms |

**Três conclusões da análise:**
1. **Quem vê o e-mail importa mais que o processador.** Só gateways ou proxies SMTP, firewalls com inspeção e servidores de e-mail têm o texto completo. Switches e roteadores simples não têm, e com TLS o conteúdo ainda vai cifrado.
2. **Volume, não latência.** Uma organização pequena recebe milhares de e-mails por dia. Mesmo 1 s por e-mail daria ≈ 86 mil por dia num único núcleo.
3. **Empacotamento.** Exportar para ONNX ou TFLite e quantizar em int8 reduz a memória em ≈ 4× e costuma acelerar 2–3× em ARM. Depois disso, a acurácia precisa ser medida de novo.

### 3.3 Perguntas de pesquisa

- **P1.** Quanto de acurácia, e de FP/FN por classe, se perde ao exportar para ONNX e quantizar (fp16, int8)?
- **P2.** Quais são a latência por e-mail (p50/p95/p99), a vazão, a memória de pico e a energia por e-mail em ARM de baixo custo, comparadas ao PC?
- **P3.** A ordem de eficiência entre os 5 modelos (TinyBERT, MiniLM, DistilBERT, MobileBERT, ALBERT) se mantém fora do PC?
- **P4.** Em que ponto da rede o detector é viável na prática?

### 3.4 Hipóteses (estimativas a verificar)

- **H1.** int8 reduz o modelo ≈ 4× (≈ 57 → 15 MB) com perda de acurácia de no máximo ~1 p.p.
- **H2.** No Raspberry Pi 4/5, a detecção fica na ordem de dezenas a centenas de ms por e-mail, suficiente para uma organização pequena.
- **H3.** O TinyBERT mantém a melhor relação acurácia/custo também na borda.

### 3.5 Método proposto

1. **Partir dos modelos já treinados** do lote oficial, sem retreinar, com o mesmo pré-processamento e o mesmo conjunto de teste (MD5 `ccfa3c3612628cdfcee16eab995a7d00`).
2. **Exportar** PyTorch → ONNX e verificar a equivalência pela diferença máxima de logits em relação ao PyTorch.
3. **Comparar variantes:** fp32, fp16 e int8 (quantização dinâmica do ONNX Runtime); TFLite é opcional.
4. **Dispositivos:**
   - PC i7, como referência;
   - Raspberry Pi 4/5, o alvo principal;
   - opcionais, conforme o hardware disponível: roteador ARM com OpenWrt, pfSense/OPNsense e Jetson.
5. **Métricas:**
   - predição: acurácia, F1, FP/FN, taxas por classe e acurácia balanceada;
   - desempenho: latência p50/p95/p99, vazão, memória de pico (RSS), tamanho em disco, tempo de carga e energia por e-mail (medidor USB/INA219);
   - RTDE recalculado na borda.
6. **Prova de conceito de integração:** detector como serviço chamado por um filtro de e-mail (milter do Postfix ou plugin do rspamd), medindo o tempo de ponta a ponta.

### 3.6 Cuidados estatísticos (lições da replicação)

- **Tamanho do teste.** Com 301 e-mails, 1 erro equivale a 0,33 p.p. e o erro-padrão binomial é ≈ ±0,5 p.p.
  - Para comparar fp32 × int8 no mesmo teste, usar o **teste de McNemar**, pareado por e-mail.
  - Considerar um teste maior, por exemplo com os 1471 phishing disponíveis, registrando um novo hash.
- **Latência.** Descartar o aquecimento, reportar a distribuição e não só a média, e registrar ou fixar a frequência da CPU, porque o Raspberry Pi reduz o clock quando esquenta.
- **Rastreabilidade.**
  - Registrar as versões (ONNX Runtime, opset, sistema operacional) e os hashes MD5.
  - Ordenar listas de arquivos com `sorted()`. O glob devolve ordens diferentes no Windows e no Linux; isso foi detectado na replicação pelos hashes MD5.

### 3.7 Contribuição esperada

- A primeira medição **real** do PHILDER em hardware limitado, que confirma ou refuta a alegação central do artigo.
- Um protocolo reprodutível para avaliar detectores de phishing baseados em transformers na borda.
- Uma discussão fundamentada de **onde** implantar o detector.

### 3.8 Riscos e limitações

- Disponibilidade de hardware.
- Dados antigos (Nazario/Enron) e teste pequeno.
- Firewalls comerciais fechados, que provavelmente exigirão pfSense/OPNsense ou um gateway Linux como substituto.

---

## 4. Conexões com as ideias já registradas no chat "Ideias dissertação Mestrado"

Ideias registradas anteriormente:
- ambiente de benchmark para modelos *tiny*, em que a pessoa informa parâmetros de recursos computacionais e a execução respeita esses parâmetros;
- backend por simulação padronizada (container, VM);
- abstração como função Python;
- interesse em aprendizado federado e em inferência privada.

**Sugestões de consolidação** (para avaliar lá; não são decisões):

1. **Complementaridade direta com o ambiente de benchmark.** A ideia de hardware real fornece a **verdade de campo** para validar o ambiente simulado.
   - Pergunta: um container limitado (CPU/memória via cgroups) prevê bem a latência e a memória medidas num Raspberry Pi real?
   - Um tema possível: *"ambiente de benchmark com restrição controlada de recursos para modelos tiny, validado contra hardware de borda real, com o PHILDER como estudo de caso"*.
   - Isso também retoma a extensão "benchmark containerizado (aproximando-se de AMP4EC)", prevista desde a Atividade 3 e o plano do artigo (Atividade 4).
2. **Ligação com aprendizado federado.** Se o detector roda na borda, cada gateway poderia **ajustar o modelo localmente** com seus próprios e-mails e compartilhar só os pesos (FL). O conteúdo dos e-mails não sai da organização. A implantação na borda é pré-requisito para isso.
3. **Ligação com inferência privada.** Detectar na borda, dentro da organização, já evita enviar e-mails a um serviço externo. É um argumento de privacidade na mesma linha.

---

## 5. Sugestão de texto para o compilado `ideias-pesquisa-mestrado.md`

> **Ideia: Detecção de phishing com modelos tiny na borda — da métrica em PC à medição real.** A replicação do PHILDER (Bispo et al., 2025) mostrou que a alegação de adequação a dispositivos limitados se apoia apenas em medições em PC/GPU, com métricas de eficiência (TES/IES/RTDE) inconsistentes no próprio artigo. Propõe-se exportar os modelos treinados para ONNX/TFLite, quantizá-los (fp16/int8) e medir acurácia, latência, vazão, memória e energia em hardware de borda real (Raspberry Pi; opcionalmente roteador OpenWrt, firewall pfSense/OPNsense, Jetson), com comparação pareada (McNemar) contra o modelo original. A ideia complementa o ambiente de benchmark com restrição controlada de recursos, servindo como verdade de campo para validá-lo, e é pré-requisito para cenários de aprendizado federado entre gateways de e-mail.

---

## 6. Status

- [x] Replicação do PHILDER e ablação concluídas (base experimental pronta).
- [x] Análise preliminar de viabilidade por tipo de equipamento (Seção 3.2).
- [ ] Consolidar no chat "Ideias dissertação Mestrado" e decidir se vira ideia própria ou parte do ambiente de benchmark.
- [ ] Discutir com o orientador.
- [ ] Levantar o hardware disponível.
- [ ] Exportação ONNX e teste de equivalência no PC (código novo; só com autorização explícita).
- [ ] Medições na borda.

### Referências de apoio

- BISPO, G. D. et al. PHILDER: Lightweight Framework for Intelligent Phishing Detection on Resource-Limited Devices. *IEEE Access*, v. 13, p. 131967–131979, 2025.
- JIAO, X. et al. TinyBERT: Distilling BERT for Natural Language Understanding. *Findings of EMNLP*, 2020.
- JACOB, B. et al. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. *CVPR*, 2018.
- PRINCE, S. J. D. *Understanding Deep Learning*. MIT Press, 2023. Cap. 8 (medição de desempenho) e cap. 12 (transformers).
