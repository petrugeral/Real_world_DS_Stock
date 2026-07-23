# Análise de Estoques: Classificação ABC-XYZ + Política de Reposição Segmentada

> **Série: Data Science no Mundo Real — Episódio 1**
> Post completo no Medium: [link]

---

## 🎯 Objetivo

Demonstrar como a segmentação **ABC-XYZ**, combinada com o cálculo de **estoque de segurança por perfil de demanda**, reduz rupturas sem aumentar proporcionalmente o capital imobilizado.

A ideia central: uma política de estoque **uniforme** (mesmo nível de serviço para todos os SKUs) trata produtos críticos e imprevisíveis da mesma forma que produtos estáveis e de baixo giro — e isso "mente" sobre onde o risco real está concentrado.

---

## 📦 Fonte dos dados

[M5 Forecasting - Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

Dados hierárquicos de vendas diárias do Walmart em 3 estados americanos (Califórnia, Texas e Wisconsin):

- **3.049 SKUs** · **10 lojas** · **1.913 dias** de histórico (2011–2016)
- Categorias: FOODS, HOBBIES, HOUSEHOLD
- Variáveis explicativas: preço semanal por loja, eventos especiais, benefício SNAP por estado

| Arquivo                      | Conteúdo                                                     |
| ---------------------------- | ------------------------------------------------------------ |
| `calendar.csv`               | Datas, semana Walmart (`wm_yr_wk`), eventos, flags SNAP      |
| `sell_prices.csv`            | Preço por item/loja/semana                                   |
| `sales_train_validation.csv` | Vendas diárias por item/loja — formato wide (`d_1`…`d_1913`) |

---

## 🧭 Estrutura do Notebook

| #   | Seção                              | O que faz                                                                                                            |
| --- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 0   | Setup                              | Imports, estilo visual, seed, `gc` para gestão de memória                                                            |
| 1   | Coleta de dados                    | Leitura dos 3 CSVs do Kaggle via path local                                                                          |
| 2   | Preparação                         | Agregação por item (sem melt total — estratégia para poupar RAM); join com preços via `wm_yr_wk`; cálculo de receita |
| 3   | Análise Exploratória               | Receita por SKU, curva de Pareto, distribuição do CV por categoria (FOODS/HOBBIES/HOUSEHOLD)                         |
| 4   | Classificação ABC-XYZ              | ABC por receita acumulada; XYZ por coeficiente de variação (CV)                                                      |
| 5   | Visualização Estratégica           | Heatmaps de qtd. de SKUs e % de receita; scatter CV × receita                                                        |
| 6   | Política de Estoque                | ROP e Safety Stock com nível de serviço diferenciado por segmento; lead time assimétrico via Monte Carlo             |
| 7   | Simulação de Impacto               | Comparativo de rupturas simuladas (12 meses): política atual × política segmentada                                   |
| 8   | Previsão de Demanda por Perfil XYZ | Modelos diferentes por classe: **SARIMA** para X, **Prophet** para Y, **Monte Carlo** para Z                         |
| 9   | Resumo Executivo                   | Diagnóstico, solução, impacto e limitações em linguagem de stakeholder                                               |

---

## 🤖 Modelos de Previsão

O notebook implementa uma estratégia de modelagem adaptativa: cada perfil de demanda recebe o modelo mais adequado ao seu comportamento.

| Classe XYZ               | Perfil                 | Modelo                         | Justificativa                                                                 |
| ------------------------ | ---------------------- | ------------------------------ | ----------------------------------------------------------------------------- |
| **X** (CV ≤ 0.20)        | Demanda estável        | SARIMA `(1,0,1)(1,1,1,7)`      | Captura autocorrelação e sazonalidade semanal com intervalo de confiança 95%  |
| **Y** (0.20 < CV ≤ 0.50) | Variabilidade moderada | Prophet                        | Lida com sazonalidades múltiplas (semanal + anual) e feriados automaticamente |
| **Z** (CV > 0.50)        | Alta imprevisibilidade | Monte Carlo (1.000 simulações) | Entrega distribuição de cenários (P10/mediana/P90) em vez de ponto único      |

O comparativo SARIMA vs. Média Móvel nos itens X mostrou que o SARIMA venceu em 4 dos 5 itens testados, com ganhos de MAE entre 0,05 e 3,73 unidades.

---

## 🧮 Metodologia de Estoque

**Ponto de Reposição (ROP):**

```
ROP = (d̄ × L) + SS
```

**Estoque de Segurança (SS):**

```
SS = z × σ_d × √L
```

Onde `d̄` = demanda diária média, `L` = lead time (dias), `σ_d` = desvio-padrão da demanda diária, `z` = z-score do nível de serviço alvo.

**Nível de serviço por segmento:**

|                       | X (estável) | Y (moderado) | Z (errático) |
| --------------------- | ----------- | ------------ | ------------ |
| **A** (receita alta)  | 95%         | 98%          | 99%          |
| **B** (receita média) | 90%         | 92%          | 95%          |
| **C** (receita baixa) | 85%         | 85%          | 90%          |

---

## ⚙️ Como Executar

### Requisitos

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels prophet scikit-learn
```

### Dados

```bash
pip install kagglehub
```

```python
import kagglehub
path = kagglehub.competition_download('m5-forecasting-accuracy')
```

Ou ajuste a variável `base` na célula de leitura para o caminho local dos arquivos.

### Notas de memória

O notebook **não faz melt do dataset completo** — operação que geraria ~57 milhões de linhas e travaria kernels com menos de 16 GB de RAM. A estratégia adotada:

- Agregação por `item_id` antes de qualquer transformação
- Join com preços via `wm_yr_wk` (chave semanal do calendário Walmart)
- `del` de DataFrames intermediários + `gc.collect()` entre etapas pesadas

---

## ⚠️ Limitações

- **Demanda censurada:** histórico com ruptura subestima a demanda real — se faltou estoque, a venda registrada é menor que a demanda verdadeira
- **Sazonalidade reclassifica SKUs:** um produto X no geral pode ser Z em dezembro; revisão trimestral das classes é recomendada
- **Lead time como distribuição:** o notebook usa Monte Carlo para o lead time na política de estoque, mas dados reais de lead time por fornecedor melhorariam a acurácia
- **Amostra para modelagem:** os modelos de previsão rodam em amostras (5–10 itens por classe) para viabilizar execução local; escalar para todos os 3.049 SKUs requer paralelização ou infraestrutura em nuvem

---

## 📚 Referências

- Chopra & Meindl — _Supply Chain Management: Strategy, Planning, and Operation_
- Hadley & Whitin — _Analysis of Inventory Systems_
- Silver, Pyke & Thomas — _Inventory and Production Management in Supply Chains_
- [Prophet](https://facebook.github.io/prophet/) — Meta, forecasting com sazonalidade múltipla
- [M5 Competition](https://mofc.unic.ac.cy/m5-competition/) — Makridakis Open Forecasting Center

---

_Série Data Science no Mundo Real — Episódio 1_
_GitHub: [link do repositório]_
