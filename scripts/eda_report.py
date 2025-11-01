"""
eda_report.py

Genera un informe exploratorio mínimo a partir del CSV limpio.

- Lee: crypto_clean_BTC_ETH_BNB.csv (por defecto)
- Escribe: EDA_summary.csv y gráficos PNG por activo

Uso:
  python eda_report.py --in crypto_clean_BTC_ETH_BNB.csv --outdir .
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Crea un resumen por activo con estadísticas descriptivas clave."""
    def agg(group: pd.DataFrame) -> pd.Series:
        g = group.sort_values("date")
        return pd.Series({
            "min_date": g["date"].min(),
            "max_date": g["date"].max(),
            "rows": len(g),
            "price_mean": g["price_usd"].mean(),
            "price_median": g["price_usd"].median(),
            "price_std": g["price_usd"].std(),
            "ret_mean": g["daily_return"].mean(skipna=True),
            "ret_std": g["daily_return"].std(skipna=True),
            "na_daily_return": int(g["daily_return"].isna().sum()),
        })

    return df.groupby(["coin_id", "symbol"], as_index=False).apply(agg).reset_index(drop=True)


def plot_price_history(df: pd.DataFrame, symbol: str, outdir: Path) -> None:
    subset = df[df["symbol"] == symbol].sort_values("date")
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=subset, x="date", y="price_usd")
    plt.title(f"Price history {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(outdir / f"price_history_{symbol}.png", dpi=150)
    plt.close()


def plot_rolling_vol(df: pd.DataFrame, symbol: str, outdir: Path) -> None:
    subset = df[df["symbol"] == symbol].sort_values("date")
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=subset, x="date", y="roll_vol_30d")
    plt.title(f"Rolling 30D volatility (annualized) {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.savefig(outdir / f"rolling_vol_30d_{symbol}.png", dpi=150)
    plt.close()


def plot_returns_hist(df: pd.DataFrame, symbol: str, outdir: Path) -> None:
    subset = df[df["symbol"] == symbol]["daily_return"].dropna()
    plt.figure(figsize=(6, 4))
    sns.histplot(subset, bins=50, kde=True)
    plt.title(f"Daily returns distribution {symbol}")
    plt.xlabel("Daily Return")
    plt.tight_layout()
    plt.savefig(outdir / f"returns_hist_{symbol}.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", default="data/crypto_clean_BTC_ETH_BNB.csv",
                    help="Ruta al CSV limpio (por defecto: data/crypto_clean_BTC_ETH_BNB.csv)")
    ap.add_argument("--outdir", default="reports/eda",
                    help="Carpeta de salida para EDA (por defecto: reports/eda)")
    args = ap.parse_args()

    # Resolver rutas relativas a la raíz del proyecto (carpeta padre de scripts/)
    ROOT = Path(__file__).resolve().parent.parent
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = (ROOT / in_path).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (ROOT / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df["date"] = pd.to_datetime(df["date"])

    # Summary table
    summary = summarize(df)
    summary.to_csv(outdir / "EDA_summary.csv", index=False)

    # Plots per ticker
    for sym in sorted(df["symbol"].unique()):
        plot_price_history(df, sym, outdir)
        plot_rolling_vol(df, sym, outdir)
        plot_returns_hist(df, sym, outdir)

    print(f"\n✅ EDA generado en {outdir.resolve()}:")
    print("- EDA_summary.csv")
    for sym in sorted(df["symbol"].unique()):
        print(f"- price_history_{sym}.png")
        print(f"- rolling_vol_30d_{sym}.png")
        print(f"- returns_hist_{sym}.png")


if __name__ == "__main__":
    main()
