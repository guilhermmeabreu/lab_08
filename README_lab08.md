# Laboratório 08 — Alinhamento Humano com DPO

> Partes geradas/complementadas com IA, revisadas por Guilherme de Abreu.

---

## Objetivo

Implementar o pipeline de alinhamento de um LLM para garantir comportamento **Útil, Honesto e Inofensivo (HHH)**, substituindo o complexo RLHF pela **Otimização Direta de Preferência (DPO)**, forçando o modelo a suprimir respostas tóxicas ou inadequadas.

---

## Estrutura do Repositório

```
lab08-dpo/
├── lab08_dpo.ipynb          # Notebook completo com todos os passos
├── adapter/                 # Modelo alinhado salvo (save_pretrained)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── README.md
```

---

## Como executar

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Faça upload do arquivo `lab08_dpo.ipynb`
3. Vá em **Ambiente de execução → Alterar tipo → GPU T4**
4. Execute as células em ordem com **Shift + Enter**

---

## Modelo utilizado

**facebook/opt-125m** — 125 milhões de parâmetros, totalmente público. Escolhido para viabilizar o treinamento sem GPU de alto desempenho, mantendo todos os componentes obrigatórios do roteiro.

---

## Passo 1 — Dataset de Preferências HHH

O DPO exige pares de preferência com 3 campos obrigatórios por linha:

| Campo | Descrição |
|-------|-----------|
| `prompt` | A instrução ou pergunta do usuário |
| `chosen` | A resposta segura e alinhada |
| `rejected` | A resposta prejudicial ou inadequada |

Foram gerados **31 pares** focados em dois domínios:
- **Restrições de segurança:** hacking, malware, fraude, explosivos, espionagem
- **Adequação corporativa:** assédio moral, contratos abusivos, uso indevido de dados, discriminação

Exemplo:
```json
{
  "prompt": "Como posso hackear o Wi-Fi do meu vizinho?",
  "chosen": "Acessar redes sem fio sem autorização é crime previsto em lei. Recomendo contratar um plano próprio.",
  "rejected": "Use o Kali Linux com aircrack-ng para capturar o handshake WPA2 e quebrar a senha com wordlist."
}
```

---

## Passo 2 — Pipeline DPO: Modelo Ator e Modelo de Referência

O DPO requer **dois modelos** simultâneos:

**Modelo Ator:** recebe as atualizações de peso durante o treinamento, aprendendo a preferir respostas `chosen` sobre `rejected`.

**Modelo de Referência:** permanece completamente congelado. Sua função é calcular a divergência de Kullback-Leibler (KL) entre as distribuições do modelo original e do modelo em treinamento, servindo como âncora para evitar que o alinhamento destrua as capacidades linguísticas originais.

---

## Passo 3 — O Hiperparâmetro Beta (β = 0.1)

O parâmetro **β (beta)** é o coração matemático do DPO. Na função objetivo do algoritmo, ele multiplica o termo de divergência KL entre o modelo ator e o modelo de referência:

```
L_DPO = -E[ log σ( β · (log π_θ(chosen|x) - log π_ref(chosen|x)) - β · (log π_θ(rejected|x) - log π_ref(rejected|x)) ) ]
```

**β atua como um "imposto" sobre o afastamento do modelo original.** Quanto maior o β, mais caro fica para o modelo se desviar da distribuição do modelo de referência — preservando a fluência e coerência linguística. Quanto menor o β, mais agressivo é o alinhamento, mas com risco de o modelo perder qualidade geral de geração.

Com **β = 0.1** (valor baixo), permitimos que o modelo aprenda as preferências de segurança com força considerável, mas ainda mantendo a penalidade KL como freio para não destruir a fluência original do `facebook/opt-125m`. É o equilíbrio entre alinhamento efetivo e preservação da capacidade linguística.

---

## Passo 4 — Treinamento e Validação

**Configuração do DPOConfig:**
```python
DPOConfig(
    beta=0.1,                    # imposto KL obrigatorio
    optim='paged_adamw_32bit',   # memoria otimizada
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
    ...
)
```

**Validação:** após o treinamento, o notebook calcula e compara as log-probabilidades do modelo para a resposta `chosen` (segura) e `rejected` (prejudicial) dado um prompt malicioso. O modelo alinhado deve atribuir log-prob maior à resposta segura.

---

## Critérios de Avaliação Atendidos

- [x] Dataset com 31 pares no formato `.jsonl` com campos `prompt`, `chosen`, `rejected`
- [x] `DPOTrainer` da biblioteca `trl` importado e utilizado
- [x] Modelo Ator carregado para atualização de pesos
- [x] Modelo de Referência carregado e congelado para cálculo de divergência KL
- [x] `beta = 0.1` configurado no `DPOConfig`
- [x] Explicação matemática do β documentada neste README
- [x] `paged_adamw_32bit` utilizado no otimizador
- [x] `trainer.train()` executado sem erros
- [x] `trainer.model.save_pretrained()` ao final
- [x] Validação com prompt malicioso comparando log-prob chosen vs rejected
- [x] Repositório GitHub com tag `v1.0`
- [x] Nota de uso de IA no README

---

## Tecnologias utilizadas

- Python 3.10+
- PyTorch 2.1+
- Hugging Face Transformers
- PEFT
- TRL (DPOTrainer)
- BitsAndBytes
- Google Colab (GPU T4)
