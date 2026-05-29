"""
main.py
-------
Orquestrador principal do projeto IBOVESPA vs S&P500.

Uso:
    python main.py
    python main.py --start 2010-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from datetime import datetime

from src.data_collection import collect_data
from src.analysis import run_analysis
from src.plots import generate_all_plots

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

import os
os.makedirs("output", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Análise comparativa IBOVESPA vs S&P500"
    )
    parser.add_argument(
        "--start",
        default="2005-01-01",
        help="Data inicial (YYYY-MM-DD). Padrão: 2005-01-01",
    )
    parser.add_argument(
        "--end",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Data final (YYYY-MM-DD). Padrão: hoje",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Diretório para salvar dados processados. Padrão: data/",
    )
    parser.add_argument(
        "--plots-dir",
        default="output/plots",
        help="Diretório para salvar gráficos. Padrão: output/plots/",
    )
    return parser.parse_args()


def print_summary(results: dict):
    """Imprime resumo executivo no terminal."""
    print("\n" + "=" * 60)
    print("  RESUMO — IBOVESPA vs S&P500")
    print("=" * 60)

    df = results["levels"]
    print(f"\n  Período analisado : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Observações       : {len(df)} dias úteis em comum")

    print("\n  ── Estatísticas dos Log-Retornos ──")
    sr = results["stats_log_returns"]
    for idx in sr.index:
        name = idx.replace("_log_ret", "")
        mu = float(sr.loc[idx, "Média"])
        sigma = float(sr.loc[idx, "Desvio-Padrão"])
        print(f"  {name:12s}  Média diária = {mu:+.5f}   σ = {sigma:.5f}")

    print("\n  ── Correlação de Pearson (Log-Retornos) ──")
    pearson = results["corr_log_ret"]["pearson"]
    cols = pearson.columns.tolist()
    if len(cols) >= 2:
        r = float(pearson.iloc[0, 1])
        print(f"  IBOVESPA × S&P500 : r = {r:.4f}")

    print("\n  ── Estacionariedade (ADF) ──")
    adf = results["adf"]
    for idx in adf.index:
        est = adf.loc[idx, "Estacionária (5%)"]
        p = float(adf.loc[idx, "p-valor"])
        print(f"  {idx:35s}  p = {p:.4f}  → {est}")

    print("\n  Gráficos salvos em: output/plots/")
    print("=" * 60 + "\n")


def main():
    args = parse_args()

    logger.info("=" * 55)
    logger.info("  IBOVESPA vs S&P500 — Pipeline iniciado")
    logger.info("=" * 55)
    logger.info(f"Período: {args.start} → {args.end}")

    # 1. Coleta de dados
    logger.info("\n[1/3] Coletando dados...")
    df_levels = collect_data(
        start=args.start,
        end=args.end,
        save_path=f"{args.data_dir}/series_diarias.csv",
    )

    # 2. Análise
    logger.info("\n[2/3] Rodando análise...")
    results = run_analysis(df_levels, output_dir=args.data_dir)

    # 3. Gráficos
    logger.info("\n[3/3] Gerando gráficos...")
    generate_all_plots(results, output_dir=args.plots_dir)

    # Resumo
    print_summary(results)
    logger.info("Pipeline concluído com sucesso.")


if __name__ == "__main__":
    main()
