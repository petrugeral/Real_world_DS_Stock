# Análise de estoques: Classificação ABC-XYZ + Política de Reposição Segmentada

> **Série: Data Science no Mundo Real — Episódio 1**
> Acompanha o post completo no Medium: [link]

## 🎯 Objetivo

Demonstrar como a segmentação **ABC-XYZ**, combinada com o cálculo de **estoque de segurança por perfil de demanda**, reduz rupturas de estoque sem aumentar proporcionalmente o capital imobilizado.

A ideia central: uma política de estoque **uniforme** (mesmo nível de serviço para todos os SKUs) trata produtos críticos e imprevisíveis da mesma forma que produtos estáveis e de baixo giro — e isso "mente" sobre onde o risco real está concentrado.

## 📦 Fonte dos dados

[M5 Forecasting - Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

Dados hierárquicos de vendas diárias do Walmart em 3 estados americanos (Califórnia, Texas e Wisconsin), cobrindo:

- **3.049 SKUs**, **10 lojas**, **1.906 dias** de histórico
- Variáveis explicativas: preço (semanal, por loja), promoções, dia da semana, eventos especiais, benefício SNAP

Arquivos utilizados:
| Arquivo | Conteúdo |
|---|---|
| `calendar.csv` | Datas de venda, semana Walmart (`wm_yr_wk`), eventos |
| `sell_prices.csv` | Preço por item/loja/semana |
| `sales_train_validation.csv` | Vendas diárias por item/loja (formato wide, `d_1`...`d_1913`) |

## 🧭 Estrutura do notebook

1. **Setup** — imports, estilo visual, seed
2. **Coleta e preparação dos dados** — join de vendas, calendário e preços; conversão para formato longo com otimização de memória (`category`, `downcast`)
3. **Análise Exploratória** — distribuição de receita por SKU, curva de Pareto, distribuição do coeficiente de variação (CV) por categoria
4. **Classificação ABC-XYZ**
   - **ABC** (importância em receita): A = top 80% acumulado · B = 80–95% · C = últimos 5%
   - **XYZ** (variabilidade da demanda via CV): X ≤ 0.20 · 0.20 < Y ≤ 0.50 · Z > 0.50
5. **Visualização Estratégica** — heatmaps de quantidade de SKUs e % de receita por segmento; scatter CV × receita
6. **Política de Estoque por Segmento** — cálculo de Ponto de Reposição (ROP) e Estoque de Segurança (SS) com nível de serviço diferenciado por classe
7. **Simulação de Impacto** — comparação de rupturas simuladas (12 meses) entre a política atual (SS fixo) e a política segmentada
8. **Resumo Executivo** — diagnóstico, solução proposta, impacto estimado e limitações, no formato para apresentação a stakeholders

## 🧮 Metodologia de estoque

**Ponto de Reposição (ROP):**

```
ROP = (d̄ × L) + SS
```

**Estoque de Segurança (SS):**

```
SS = z × σ_d × √L
```

Onde `d̄` é a demanda diária média, `L` o lead time em dias, `σ_d` o desvio-padrão da demanda diária e `z` o z-score associado ao nível de serviço desejado.

**A inovação da abordagem:** o nível de serviço (`z`) não é fixo — varia por segmento ABC-XYZ, do mais conservador (99% para AZ — alta receita, alta variabilidade) ao mais tolerante a risco (85% para CX — baixa receita, demanda estável ou moderada):

|                       | X (estável) | Y (moderado) | Z (errático) |
| --------------------- | ----------- | ------------ | ------------ |
| **A** (receita alta)  | 95%         | 98%          | 99%          |
| **B** (receita média) | 90%         | 92%          | 95%          |
| **C** (receita baixa) | 85%         | 85%          | 90%          |

## ⚙️ Como executar

### Requisitos

```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Dados

1. Baixe o dataset no Kaggle: [m5-forecasting-accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
2. Ajuste a variável `base` (célula de leitura dos dados) para o caminho local dos arquivos `.csv`

### Execução

Rode o notebook célula a célula, na ordem — as etapas de agregação (item 2) são custosas em memória; o pipeline processa loja a loja e descarta DataFrames intermediários (`del`) para conter o uso de RAM.

## ⚠️ Limitações conhecidas

- **Lead time tratado como fixo** (7 dias hipotéticos) — na prática, é uma distribuição, muitas vezes assimétrica
- **Histórico com ruptura subestima a demanda real** (demanda censurada — se faltou estoque, a venda registrada não reflete a demanda verdadeira)
- **Sazonalidade pode reclassificar SKUs** ao longo do ano — recomenda-se revisão periódica (trimestral) das classes ABC-XYZ
- **O modelo não substitui julgamento operacional** — é uma ferramenta de apoio à decisão, não um piloto automático

## 📚 Referências

- Chopra & Meindl — _Supply Chain Management: Strategy, Planning, and Operation_
- Hadley & Whitin — _Analysis of Inventory Systems_
- Silver, Pyke & Thomas — _Inventory and Production Management in Supply Chains_
- [Prophet](https://facebook.github.io/prophet/) — forecasting para séries temporais com sazonalidade

---

_Série Data Science no Mundo Real — Episódio 1_
_Post completo: [link do Medium]_
