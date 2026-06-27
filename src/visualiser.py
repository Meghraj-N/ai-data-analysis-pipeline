"""
src/visualiser.py
────────────────────────────────────────────────────────────────
Dashboard & Chart Generation Module
Author : Meghraj Nikalje | B.Sc. Data Science, Mumbai University
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from src.logger import get_logger

log = get_logger(__name__)


class Visualiser:
    """Generates a multi-panel analytics dashboard and individual charts."""

    # Colour palette
    BLUE   = "#1A56A0"
    GREEN  = "#27AE60"
    RED    = "#E74C3C"
    PURPLE = "#8E44AD"
    ORANGE = "#E67E22"
    GRAY   = "#7F8C8D"
    PALETTE = [BLUE, GREEN, RED, PURPLE, ORANGE, GRAY, "#F39C12", "#1ABC9C"]

    def __init__(self, df: pd.DataFrame, stats: dict, config: Optional[dict] = None):
        self.df     = df
        self.stats  = stats
        self.config = config or {}
        self.dpi    = int(self.config.get("figure_dpi", 150))
        sns.set_theme(style=self.config.get("style", "whitegrid"), palette=self.PALETTE)
        log.info("Visualiser initialised.")

    # ── helpers ────────────────────────────────────────────────
    def _fmt_inr(self, x, _):
        return f"₹{x:,.0f}"

    def _save(self, fig: plt.Figure, name: str, output_dir: str) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = str(Path(output_dir) / name)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info("Saved chart → %s", path)
        return path

    # ── Main Dashboard (6 panels) ──────────────────────────────
    def build_dashboard(self, output_dir: str = "report") -> str:
        fig = plt.figure(figsize=(18, 12), facecolor="white")
        fig.suptitle("Sales Analytics Dashboard", fontsize=20, fontweight="bold",
                     color=self.BLUE, y=1.01)

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        self._plot_revenue_by_category(ax1)
        self._plot_order_status_pie(ax2)
        self._plot_monthly_trend(ax3)
        self._plot_sales_rep(ax4)
        self._plot_revenue_distribution(ax5)
        self._plot_region_heatmap(ax6)

        return self._save(fig, "dashboard.png", output_dir)

    # ── Panel 1: Revenue by Category ──────────────────────────
    def _plot_revenue_by_category(self, ax: plt.Axes) -> None:
        rev = self.stats.get("revenue_by_category", {}).get("total", {})
        if not rev:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            return

        cats = list(rev.keys())
        vals = list(rev.values())
        bars = ax.bar(cats, vals, color=self.PALETTE[:len(cats)], edgecolor="white", linewidth=0.8)
        ax.set_title("Revenue by Category", fontweight="bold", pad=10, color=self.BLUE)
        ax.set_ylabel("Total Revenue")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._fmt_inr))
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"₹{val:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Panel 2: Order Status Pie ──────────────────────────────
    def _plot_order_status_pie(self, ax: plt.Axes) -> None:
        counts = self.stats.get("order_status", {}).get("counts", {})
        if not counts:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            return

        labels = list(counts.keys())
        sizes  = list(counts.values())
        colors = [self.GREEN if l == "Completed" else self.RED if l == "Cancelled" else self.ORANGE
                  for l in labels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%", colors=colors,
            startangle=90, pctdistance=0.82,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight("bold")
        ax.set_title("Order Status Distribution", fontweight="bold", pad=10, color=self.BLUE)

    # ── Panel 3: Monthly Revenue Trend ────────────────────────
    def _plot_monthly_trend(self, ax: plt.Axes) -> None:
        trend = self.stats.get("monthly_trend", {}).get("revenue", {})
        if not trend:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            return

        months = list(trend.keys())
        values = list(trend.values())
        ax.plot(months, values, marker="o", color=self.BLUE, lw=2.5, ms=8, zorder=3)
        ax.fill_between(months, values, alpha=0.12, color=self.BLUE)

        # Annotate max point
        max_idx = values.index(max(values))
        ax.annotate(f"Peak\n₹{values[max_idx]:,.0f}",
                    xy=(months[max_idx], values[max_idx]),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8, color=self.BLUE, fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": self.BLUE, "lw": 0.8})

        ax.set_title("Monthly Revenue Trend", fontweight="bold", pad=10, color=self.BLUE)
        ax.set_ylabel("Revenue")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._fmt_inr))
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 4: Top Sales Reps ────────────────────────────────
    def _plot_sales_rep(self, ax: plt.Axes) -> None:
        rep = self.stats.get("sales_rep_performance", {}).get("total", {})
        if not rep:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            return

        sorted_rep = dict(sorted(rep.items(), key=lambda x: x[1], reverse=True)[:6])
        reps  = list(sorted_rep.keys())
        vals  = list(sorted_rep.values())
        colors = [self.GREEN if i == 0 else self.BLUE for i in range(len(reps))]
        bars = ax.barh(reps, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title("Sales Rep Performance", fontweight="bold", pad=10, color=self.BLUE)
        ax.set_xlabel("Total Revenue")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(self._fmt_inr))
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                    f"₹{val:,.0f}", va="center", fontsize=8)

    # ── Panel 5: Revenue Distribution (histogram + KDE) ───────
    def _plot_revenue_distribution(self, ax: plt.Axes) -> None:
        if "total_revenue" not in self.df.columns:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            return

        data = self.df["total_revenue"].dropna()
        ax.hist(data, bins=12, color=self.PURPLE, alpha=0.65, edgecolor="white", linewidth=0.8, density=True)

        # KDE overlay
        from scipy.stats import gaussian_kde
        xs  = np.linspace(data.min(), data.max(), 200)
        kde = gaussian_kde(data)
        ax.plot(xs, kde(xs), color=self.PURPLE, lw=2.5, label="KDE")

        # Mean / Median lines
        ax.axvline(data.mean(),   color=self.RED,   lw=1.5, ls="--", label=f"Mean ₹{data.mean():,.0f}")
        ax.axvline(data.median(), color=self.GREEN, lw=1.5, ls=":",  label=f"Median ₹{data.median():,.0f}")

        ax.set_title("Order Value Distribution", fontweight="bold", pad=10, color=self.BLUE)
        ax.set_xlabel("Order Value (₹)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 6: Region Revenue Heatmap ───────────────────────
    def _plot_region_heatmap(self, ax: plt.Axes) -> None:
        if not {"region", "category", "total_revenue"}.issubset(self.df.columns):
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            return

        pivot = self.df.pivot_table(
            values="total_revenue", index="region", columns="category", aggfunc="sum", fill_value=0
        )
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="Blues",
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"format": "₹%.0f", "shrink": 0.8})
        ax.set_title("Revenue Heatmap (Region × Category)", fontweight="bold", pad=10, color=self.BLUE)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)
