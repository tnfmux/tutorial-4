"""
plots.py
--------
Geração de todos os gráficos comparativos entre IBOVESPA e S&P500.
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# ─── Estilo global ──────────────────────────────────────────────────
COLORS = {
    "IBOVESPA": "#009C3B",    # verde Brasil
    "SP500": "#002868",       # azul EUA
    "accent": "#FEDF00",      # amarelo Brasil (destaques)
    "gray": "#888888",
    "bg": "#F8F8F8",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": "#FFFFFF",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

CRISIS_PERIODS = [
    ("2008-09-01", "2009-06-01", "Crise 2008"),
    ("2020-02-01", "2020-06-01", "COVID-19"),
]


def _add_crisis_bands(ax, df_index):
    """Adiciona bandas cinzas para períodos de crise."""
    for start, end, label in CRISIS_PERIODS:
        try:
            ax.axvspan(
                pd.Timestamp(start), pd.Timestamp(end),
                alpha=0.12, color="red", label=label
            )
        except Exception:
            pass


def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gráfico salvo: {path}")


# ─────────────────────────────────────────────────────────────────────
# 1. SÉRIES HISTÓRICAS (NÍVEL)
# ─────────────────────────────────────────────────────────────────────

def plot_series_levels(df: pd.DataFrame, output_dir: str = "output/plots"):
    """Gráfico das séries em nível (eixos duplos)."""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.suptitle("IBOVESPA vs S&P500 — Série Histórica (Nível)", fontweight="bold", y=1.01)

    color_ibov = COLORS["IBOVESPA"]
    color_sp = COLORS["SP500"]

    ax1.plot(df.index, df["IBOVESPA"], color=color_ibov, linewidth=1.2, label="IBOVESPA")
    ax1.set_ylabel("IBOVESPA (pontos)", color=color_ibov)
    ax1.tick_params(axis="y", labelcolor=color_ibov)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["SP500"], color=color_sp, linewidth=1.2, linestyle="--", label="S&P500")
    ax2.set_ylabel("S&P500 (pontos)", color=color_sp)
    ax2.tick_params(axis="y", labelcolor=color_sp)

    _add_crisis_bands(ax1, df.index)

    # Legenda unificada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    _savefig(fig, f"{output_dir}/1_series_nivel.png")


# ─────────────────────────────────────────────────────────────────────
# 2. SÉRIE NORMALIZADA (BASE 100)
# ─────────────────────────────────────────────────────────────────────

def plot_series_normalized(df_norm: pd.DataFrame, output_dir: str = "output/plots"):
    """Gráfico das séries normalizadas na base 100."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("IBOVESPA vs S&P500 — Índice Base 100", fontweight="bold")

    for col in df_norm.columns:
        color = COLORS.get(col, COLORS["gray"])
        ax.plot(df_norm.index, df_norm[col], color=color, linewidth=1.4, label=col)

    _add_crisis_bands(ax, df_norm.index)
    ax.axhline(100, color=COLORS["gray"], linestyle=":", linewidth=0.8)
    ax.set_ylabel("Índice (base 100)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    _savefig(fig, f"{output_dir}/2_series_base100.png")


# ─────────────────────────────────────────────────────────────────────
# 3. RETORNOS DIÁRIOS
# ─────────────────────────────────────────────────────────────────────

def plot_returns(returns: pd.DataFrame, output_dir: str = "output/plots"):
    """Gráfico dos retornos diários (simples e log)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle("Retornos Diários — IBOVESPA vs S&P500", fontweight="bold")

    pairs = [
        ("IBOVESPA_ret_simples", "SP500_ret_simples", "Retorno Simples"),
        ("IBOVESPA_log_ret", "SP500_log_ret", "Log-Retorno"),
    ]

    for row_idx, (col_ibov, col_sp, label) in enumerate(pairs):
        for col_idx, (col, name) in enumerate([(col_ibov, "IBOVESPA"), (col_sp, "S&P500")]):
            ax = axes[row_idx][col_idx]
            color = COLORS[name]
            ax.plot(returns.index, returns[col], color=color, linewidth=0.6, alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_title(f"{label} — {name}")
            ax.set_ylabel("Retorno")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.autofmt_xdate()
    plt.tight_layout()
    _savefig(fig, f"{output_dir}/3_retornos_diarios.png")


# ─────────────────────────────────────────────────────────────────────
# 4. HISTOGRAMAS DOS RETORNOS
# ─────────────────────────────────────────────────────────────────────

def plot_return_histograms(returns: pd.DataFrame, output_dir: str = "output/plots"):
    """Histogramas com curva normal sobreposta para cada série de retornos."""
    cols_to_plot = [c for c in returns.columns if "log_ret" in c]
    fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(14, 5))
    fig.suptitle("Distribuição dos Log-Retornos Diários", fontweight="bold")

    if len(cols_to_plot) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_to_plot):
        name = "IBOVESPA" if "IBOVESPA" in col else "SP500"
        color = COLORS[name]
        data = returns[col].dropna()

        ax.hist(data, bins=80, color=color, alpha=0.6, density=True, label=name)

        # Curva normal
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color="black", linewidth=1.5,
                linestyle="--", label="Normal")

        ax.set_title(f"{name}\nμ={mu:.4f}  σ={sigma:.4f}")
        ax.set_xlabel("Log-Retorno")
        ax.set_ylabel("Densidade")
        ax.legend()

    plt.tight_layout()
    _savefig(fig, f"{output_dir}/4_histogramas_retornos.png")


# ─────────────────────────────────────────────────────────────────────
# 5. SCATTER PLOT — CORRELAÇÃO
# ─────────────────────────────────────────────────────────────────────

def plot_scatter_correlation(df: pd.DataFrame, title: str, filename: str, output_dir: str):
    """Scatter plot entre duas séries com linha de regressão."""
    cols = df.columns.tolist()
    if len(cols) < 2:
        return

    x_col, y_col = cols[0], cols[1]
    x = df[x_col].dropna()
    y = df[y_col].dropna()
    common = x.index.intersection(y.index)
    x, y = x[common], y[common]

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.25, s=8, color=COLORS["IBOVESPA"])
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color=COLORS["SP500"],
            linewidth=2, label=f"y = {slope:.4f}x + {intercept:.2f}\nR² = {r_value**2:.4f}  p = {p_value:.4f}")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    _savefig(fig, f"{output_dir}/{filename}")


def plot_all_scatter(df_levels: pd.DataFrame, returns: pd.DataFrame, output_dir: str = "output/plots"):
    """Gera scatter plots para níveis e retornos."""
    plot_scatter_correlation(
        df_levels, "Correlação IBOVESPA vs S&P500 — Níveis",
        "5a_scatter_niveis.png", output_dir
    )
    ret_simples = returns[[c for c in returns.columns if "ret_simples" in c]]
    plot_scatter_correlation(
        ret_simples, "Correlação IBOVESPA vs S&P500 — Retornos Simples",
        "5b_scatter_ret_simples.png", output_dir
    )
    log_ret = returns[[c for c in returns.columns if "log_ret" in c]]
    plot_scatter_correlation(
        log_ret, "Correlação IBOVESPA vs S&P500 — Log-Retornos",
        "5c_scatter_log_ret.png", output_dir
    )


# ─────────────────────────────────────────────────────────────────────
# 6. HEATMAP DE CORRELAÇÃO
# ─────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(corr_dict: dict, title: str, filename: str, output_dir: str):
    """Heatmap da matriz de correlação de Pearson."""
    pearson = corr_dict["pearson"]
    pvals = corr_dict["pearson_pvalues"]

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.zeros_like(pearson, dtype=bool)

    annot = pearson.round(4).astype(str)
    for i in range(len(pearson)):
        for j in range(len(pearson.columns)):
            p = pvals.iloc[i, j]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            annot.iloc[i, j] = f"{float(pearson.iloc[i,j]):.4f}{stars}"

    sns.heatmap(pearson, annot=annot, fmt="", cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, f"{output_dir}/{filename}")


def plot_all_heatmaps(corr_levels, corr_ret_simples, corr_log_ret, output_dir: str = "output/plots"):
    plot_correlation_heatmap(corr_levels, "Correlação — Níveis", "6a_heatmap_niveis.png", output_dir)
    plot_correlation_heatmap(corr_ret_simples, "Correlação — Retornos Simples", "6b_heatmap_ret_simples.png", output_dir)
    plot_correlation_heatmap(corr_log_ret, "Correlação — Log-Retornos", "6c_heatmap_log_ret.png", output_dir)


# ─────────────────────────────────────────────────────────────────────
# 7. ROLLING CORRELATION
# ─────────────────────────────────────────────────────────────────────

def plot_rolling_correlation(returns: pd.DataFrame, window: int = 63, output_dir: str = "output/plots"):
    """
    Correlação móvel (rolling) de 63 dias úteis (~3 meses)
    entre os log-retornos do IBOVESPA e S&P500.
    """
    ibov = returns["IBOVESPA_log_ret"]
    sp = returns["SP500_log_ret"]

    rolling_corr = ibov.rolling(window).corr(sp)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rolling_corr.index, rolling_corr, color=COLORS["SP500"], linewidth=1.2)
    ax.axhline(0, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.fill_between(rolling_corr.index, rolling_corr, 0,
                    where=(rolling_corr > 0), alpha=0.2, color=COLORS["IBOVESPA"])
    ax.fill_between(rolling_corr.index, rolling_corr, 0,
                    where=(rolling_corr < 0), alpha=0.2, color="red")

    ax.set_title(f"Correlação Móvel (janela = {window} dias úteis) — Log-Retornos", fontweight="bold")
    ax.set_ylabel("Correlação de Pearson")
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    _savefig(fig, f"{output_dir}/7_rolling_correlation.png")


# ─────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────

def generate_all_plots(results: dict, output_dir: str = "output/plots"):
    """
    Gera todos os gráficos a partir do dicionário de resultados da análise.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_series_levels(results["levels"], output_dir)
    plot_series_normalized(results["normalized"], output_dir)
    plot_returns(results["returns"], output_dir)
    plot_return_histograms(results["returns"], output_dir)
    plot_all_scatter(results["levels"], results["returns"], output_dir)
    plot_all_heatmaps(
        results["corr_levels"],
        results["corr_ret_simples"],
        results["corr_log_ret"],
        output_dir,
    )
    plot_rolling_correlation(results["returns"], output_dir=output_dir)

    logger.info(f"Todos os gráficos gerados em: {output_dir}")


if __name__ == "__main__":
    print("plots.py — execute via main.py para gerar os gráficos.")


# ─────────────────────────────────────────────────────────────────────
# 8. GRÁFICO DE BARRAS — RESUMO DE CORRELAÇÕES
# ─────────────────────────────────────────────────────────────────────

def plot_correlation_summary(corr_summary: "pd.DataFrame", output_dir: str = "output/plots"):
    """
    Gráfico de barras comparando Pearson e Spearman
    nos três contextos: nível, retorno simples, log-retorno.
    """
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Correlação IBOVESPA vs S&P500 por Contexto", fontweight="bold")

    labels = corr_summary.index.tolist()
    x = range(len(labels))
    width = 0.35

    pearson  = corr_summary["Pearson r"].tolist()
    spearman = corr_summary["Spearman r"].tolist()

    bars1 = ax.bar([i - width/2 for i in x], pearson,  width, label="Pearson",  color=COLORS["IBOVESPA"], alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], spearman, width, label="Spearman", color=COLORS["SP500"],    alpha=0.85)

    # Valores em cima das barras
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coeficiente de Correlação")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    plt.tight_layout()
    _savefig(fig, f"{output_dir}/8_correlacao_resumo.png")
