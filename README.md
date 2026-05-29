# ibovespa-sp500

Análise comparativa entre IBOVESPA e S&P500 — séries históricas, retornos e correlações.

---

## O que faz

- Baixa dados diários do IBOVESPA (Yahoo Finance) e S&P500 (FRED)
- Calcula retornos simples e log-retornos
- Gera estatísticas descritivas e testa estacionariedade (ADF)
- Plota séries, histogramas, scatter de correlação e correlação móvel
- Discute correlação espúria entre séries em nível vs. retornos

---

## Estrutura

```
ibovespa_sp500/
├── src/
│   ├── data_collection.py   # coleta e alinhamento das séries
│   ├── analysis.py          # estatísticas, retornos, ADF, correlações
│   └── plots.py             # geração dos gráficos
├── data/                    # CSVs gerados em runtime
├── output/plots/            # gráficos exportados
├── main.py                  # entry point
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/tnfmux/ibovespa_sp500.git
cd ibovespa_sp500

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Crie um `.env` na raiz com sua chave do FRED (gratuita em [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)):

```
FRED_API_KEY=sua_chave_aqui
```

---

## Uso

```bash
python main.py
python main.py --start 2010-01-01 --end 2024-12-31
```

Outputs gerados em `data/` (CSVs) e `output/plots/` (PNGs).

---

## Dependências principais

| lib | uso |
|---|---|
| `yfinance` | coleta IBOVESPA |
| `fredapi` | coleta S&P500 |
| `pandas` / `numpy` | manipulação |
| `matplotlib` / `seaborn` | visualizações |
| `statsmodels` | ADF, econometria |
| `scipy` | testes estatísticos |

---

## Fontes

- FRED: https://fred.stlouisfed.org/
- Yahoo Finance: https://finance.yahoo.com/

---

## Autor

[tnfmux](https://github.com/tnfmux)
