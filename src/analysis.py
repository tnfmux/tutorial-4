"""
analysis.py
-----------
Estatísticas descritivas, cálculo de retornos e análise de correlação
entre IBOVESPA e S&P500.
"""

import os
import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. RETORNOS
# ─────────────────────────────────────────────

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos simples e log-retornos para cada coluna do DataFrame.

    Retorno simples  : r_t = (P_t / P_{t-1}) - 1
    Log-retorno      : r_t = ln(P_t / P_{t-1})

    Returns
    -------
    pd.DataFrame com colunas:
        IBOVESPA_ret_simples, SP500_ret_simples,
        IBOVESPA_log_ret,     SP500_log_ret
    """
    ret = pd.DataFrame(index=df.index)

    for col in df.columns:
        ret[f"{col}_ret_simples"] = df[col].pct_change()
        ret[f"{col}_log_ret"] = np.log(df[col] / df[col].shift(1))

    ret = ret.dropna()
    logger.info(f"Retornos calculados: {len(ret)} observações.")
    return ret


# ─────────────────────────────────────────────
# 2. ESTATÍSTICAS DESCRITIVAS
# ─────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Calcula estatísticas descritivas detalhadas para um DataFrame.

    Inclui: média, mediana, desvio-padrão, variância, mínimo, máximo,
    assimetria (skewness), curtose, e JB test de normalidade.
    """
    results = {}

    for col in df.columns:
        s = df[col].dropna()
        jb_stat, jb_p = stats.jarque_bera(s)

        results[col] = {
            "N": len(s),
            "Média": s.mean(),
            "Mediana": s.median(),
            "Desvio-Padrão": s.std(),
            "Variância": s.var(),
            "Mínimo": s.min(),
            "Máximo": s.max(),
            "Assimetria": s.skew(),
            "Curtose": s.kurtosis(),  # excess kurtosis (normal = 0)
            "JB Stat": jb_stat,
            "JB p-valor": jb_p,
            "Normal (5%)": "Sim" if jb_p > 0.05 else "Não",
        }

    out = pd.DataFrame(results).T
    if label:
        logger.info(f"Estatísticas descritivas — {label}")
    return out


# ─────────────────────────────────────────────
# 3. TESTE ADF (ESTACIONARIEDADE)
# ─────────────────────────────────────────────

def adf_test(series: pd.Series, name: str = "") -> dict:
    """
    Teste Augmented Dickey-Fuller para verificar estacionariedade.

    H0: série possui raiz unitária (não-estacionária)
    Rejeita H0 (série estacionária) se p-valor < 0.05
    """
    result = adfuller(series.dropna(), autolag="AIC")
    out = {
        "Série": name or series.name,
        "ADF Stat": result[0],
        "p-valor": result[1],
        "Lags usados": result[2],
        "N": result[3],
        "Estacionária (5%)": "Sim" if result[1] < 0.05 else "Não",
        "Valor Crítico 1%": result[4]["1%"],
        "Valor Crítico 5%": result[4]["5%"],
        "Valor Crítico 10%": result[4]["10%"],
    }
    return out


def run_adf_tests(df_levels: pd.DataFrame, df_returns: pd.DataFrame) -> pd.DataFrame:
    """Roda ADF para todos os níveis e retornos."""
    rows = []
    for col in df_levels.columns:
        rows.append(adf_test(df_levels[col], name=f"{col} (nível)"))
    for col in df_returns.columns:
        rows.append(adf_test(df_returns[col], name=col))
    return pd.DataFrame(rows).set_index("Série")


# ─────────────────────────────────────────────
# 4. CORRELAÇÃO
# ─────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame, label: str = "") -> dict:
    """
    Calcula correlação de Pearson e Spearman entre todas as colunas.

    Returns
    -------
    dict com 'pearson' e 'spearman' DataFrames, e 'pearson_pvalues'
    """
    cols = df.columns.tolist()
    n = len(cols)

    pearson_r = np.zeros((n, n))
    pearson_p = np.zeros((n, n))
    spearman_r = np.zeros((n, n))

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            s1 = df[c1].dropna()
            s2 = df[c2].dropna()
            common = s1.index.intersection(s2.index)
            r, p = stats.pearsonr(s1[common], s2[common])
            sp, _ = stats.spearmanr(s1[common], s2[common])
            pearson_r[i, j] = r
            pearson_p[i, j] = p
            spearman_r[i, j] = sp

    result = {
        "pearson": pd.DataFrame(pearson_r, index=cols, columns=cols),
        "pearson_pvalues": pd.DataFrame(pearson_p, index=cols, columns=cols),
        "spearman": pd.DataFrame(spearman_r, index=cols, columns=cols),
    }

    if label:
        logger.info(f"Correlação ({label}) — Pearson:\n{result['pearson'].round(4)}")

    return result


# ─────────────────────────────────────────────
# 5. NORMALIZAÇÃO BASE 100
# ─────────────────────────────────────────────

def normalize_base100(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza cada série para base 100 na primeira observação."""
    return df.div(df.iloc[0]) * 100


# ─────────────────────────────────────────────
# 6. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def run_analysis(
    df_levels: pd.DataFrame,
    output_dir: str = "data",
) -> dict:
    """
    Executa toda a análise e salva CSVs.

    Parameters
    ----------
    df_levels  : DataFrame com séries em nível (IBOVESPA, SP500)
    output_dir : diretório para salvar resultados

    Returns
    -------
    dict com todos os resultados intermediários
    """
    os.makedirs(output_dir, exist_ok=True)

    # Retornos
    returns = compute_returns(df_levels)
    returns.to_csv(f"{output_dir}/retornos.csv")

    # Normalização
    normalized = normalize_base100(df_levels)

    # Estatísticas descritivas
    stats_levels = descriptive_stats(df_levels, label="Níveis")
    stats_returns_simples = descriptive_stats(
        returns[[c for c in returns.columns if "ret_simples" in c]],
        label="Retornos Simples",
    )
    stats_log_returns = descriptive_stats(
        returns[[c for c in returns.columns if "log_ret" in c]],
        label="Log-Retornos",
    )

    all_stats = pd.concat(
        [stats_levels, stats_returns_simples, stats_log_returns],
        keys=["Níveis", "Retorno Simples", "Log-Retorno"],
    )
    all_stats.to_csv(f"{output_dir}/estatisticas_descritivas.csv")

    # ADF
    adf_results = run_adf_tests(df_levels, returns)
    adf_results.to_csv(f"{output_dir}/adf_tests.csv")

    # Correlações
    corr_levels = correlation_analysis(df_levels, label="Níveis")
    corr_ret_simples = correlation_analysis(
        returns[[c for c in returns.columns if "ret_simples" in c]],
        label="Retornos Simples",
    )
    corr_log_ret = correlation_analysis(
        returns[[c for c in returns.columns if "log_ret" in c]],
        label="Log-Retornos",
    )

    # Salvar correlações
    corr_levels["pearson"].to_csv(f"{output_dir}/corr_niveis_pearson.csv")
    corr_ret_simples["pearson"].to_csv(f"{output_dir}/corr_ret_simples_pearson.csv")
    corr_log_ret["pearson"].to_csv(f"{output_dir}/corr_log_ret_pearson.csv")

    logger.info("Análise concluída. CSVs salvos em: " + output_dir)

    return {
        "levels": df_levels,
        "normalized": normalized,
        "returns": returns,
        "stats_levels": stats_levels,
        "stats_returns_simples": stats_returns_simples,
        "stats_log_returns": stats_log_returns,
        "adf": adf_results,
        "corr_levels": corr_levels,
        "corr_ret_simples": corr_ret_simples,
        "corr_log_ret": corr_log_ret,
    }


if __name__ == "__main__":
    # Teste rápido com dados sintéticos
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range("2010-01-01", periods=500, freq="B")
    fake = pd.DataFrame({
        "IBOVESPA": np.cumsum(np.random.randn(500)) + 50000,
        "SP500": np.cumsum(np.random.randn(500)) + 3000,
    }, index=dates)

    results = run_analysis(fake)
    print("\n=== Estatísticas Descritivas (Níveis) ===")
    print(results["stats_levels"].to_string())
    print("\n=== ADF Tests ===")
    print(results["adf"].to_string())


# ─────────────────────────────────────────────────────────────────────
# 7. TABELA RESUMO DE CORRELAÇÕES
# ─────────────────────────────────────────────────────────────────────

def correlation_summary_table(
    corr_levels: dict,
    corr_ret_simples: dict,
    corr_log_ret: dict,
    output_dir: str = "data",
) -> pd.DataFrame:
    """
    Monta uma tabela resumo com Pearson, Spearman e p-valor
    para cada contexto (nível, retorno simples, log-retorno).
    Salva em CSV e imprime no terminal.
    """
    def extract(corr_dict, label):
        cols = corr_dict["pearson"].columns.tolist()
        if len(cols) < 2:
            return {}
        c1, c2 = cols[0], cols[1]
        return {
            "Contexto": label,
            "Pearson r": round(float(corr_dict["pearson"].loc[c1, c2]), 4),
            "Pearson p-valor": round(float(corr_dict["pearson_pvalues"].loc[c1, c2]), 6),
            "Spearman r": round(float(corr_dict["spearman"].loc[c1, c2]), 4),
            "Significativo (5%)": "Sim" if float(corr_dict["pearson_pvalues"].loc[c1, c2]) < 0.05 else "Não",
        }

    rows = [
        extract(corr_levels,      "Nível"),
        extract(corr_ret_simples, "Retorno Simples"),
        extract(corr_log_ret,     "Log-Retorno"),
    ]

    df = pd.DataFrame(rows).set_index("Contexto")

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/correlacao_resumo.csv")

    print("\n" + "=" * 55)
    print("  TESTES DE CORRELAÇÃO — IBOVESPA vs S&P500")
    print("=" * 55)
    print(df.to_string())
    print("=" * 55 + "\n")

    return df
