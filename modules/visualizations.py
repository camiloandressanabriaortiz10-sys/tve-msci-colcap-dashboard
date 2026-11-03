from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .risk_metrics import gpd_cdf, gpd_pdf


COLORS = {"navy": "#0A2647", "blue": "#1C5D99", "cyan": "#2DA9C8", "gold": "#D9A441", "red": "#C6424A", "gray": "#6B7280", "light": "#E9F1F7"}


def style(fig: go.Figure, title: str, height: int = 470) -> go.Figure:
    fig.update_layout(title=title, template="plotly_white", height=height, margin=dict(l=12, r=12, t=55, b=12), hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def loss_timeline(data: pd.DataFrame, threshold: float, var_value: float, confidence: float) -> go.Figure:
    fig = go.Figure()
    for sample, color in [("Estimación", COLORS["blue"]), ("Validación", COLORS["cyan"])]:
        frame = data[data["muestra"] == sample]
        fig.add_trace(go.Scatter(x=frame["fecha"], y=frame["perdida"], mode="lines", name=sample, line=dict(color=color, width=1.25), customdata=np.column_stack([frame["retorno"], frame["muestra"]]), hovertemplate="Fecha: %{x|%d/%m/%Y}<br>Pérdida: %{y:.2%}<br>Retorno: %{customdata[0]:.2%}<br>Muestra: %{customdata[1]}<extra></extra>"))
    exceptions = data[data["perdida"] > var_value]
    fig.add_trace(go.Scatter(x=exceptions["fecha"], y=exceptions["perdida"], mode="markers", name=f"Excepciones VaR {confidence:.1%}", marker=dict(color=COLORS["red"], size=7, symbol="diamond"), hovertemplate="Excepción<br>Fecha: %{x|%d/%m/%Y}<br>Pérdida: %{y:.2%}<extra></extra>"))
    fig.add_hline(y=threshold, line_dash="dot", line_color=COLORS["gold"], annotation_text="Umbral POT")
    fig.add_hline(y=var_value, line_dash="dash", line_color=COLORS["red"], annotation_text=f"VaR TVE {confidence:.1%}")
    fig.update_yaxes(tickformat=".1%", title="Pérdida logarítmica diaria")
    return style(fig, "Pérdidas diarias y referencias del modelo POT–GPD", 540)


def gpd_figure(losses: np.ndarray, threshold: float, xi: float, sigma: float) -> go.Figure:
    excesses = np.sort(losses[losses > threshold] - threshold)
    y = np.linspace(0, max(excesses) * 1.03, 300)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Excesos sobre el umbral", "Ajuste acumulado GPD"))
    fig.add_trace(go.Histogram(x=excesses, histnorm="probability density", marker_color=COLORS["cyan"], opacity=.65, name="Excesos observados"), row=1, col=1)
    fig.add_trace(go.Scatter(x=y, y=gpd_pdf(y, xi, sigma), mode="lines", line=dict(color=COLORS["navy"], width=3), name="Densidad GPD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=excesses, y=np.arange(1, len(excesses)+1)/len(excesses), mode="markers", marker=dict(color=COLORS["blue"], size=5), name="Empírica"), row=1, col=2)
    fig.add_trace(go.Scatter(x=y, y=gpd_cdf(y, xi, sigma), mode="lines", line=dict(color=COLORS["red"], width=3), name="GPD ajustada"), row=1, col=2)
    fig.update_xaxes(tickformat=".1%", title="Exceso y = L − u")
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    return style(fig, "Distribución Generalizada de Pareto sobre los excesos", 440)


def threshold_figure(table: pd.DataFrame, chosen: float) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=table["percentil"], y=table["exceso_medio"], mode="lines+markers", name="Exceso medio", line=dict(color=COLORS["blue"])), secondary_y=False)
    fig.add_trace(go.Bar(x=table["percentil"], y=table["nu"], name="Excedencias", marker_color=COLORS["gold"], opacity=.7), secondary_y=True)
    fig.add_vline(x=chosen, line_dash="dash", line_color=COLORS["red"], annotation_text="p90 seleccionado")
    fig.update_xaxes(tickformat=".0%", title="Percentil candidato")
    fig.update_yaxes(tickformat=".2%", title_text="Exceso medio", secondary_y=False)
    fig.update_yaxes(title_text="Número de excedencias", secondary_y=True)
    return style(fig, "Selección del umbral", 430)


def risk_comparison(var_es: pd.DataFrame) -> go.Figure:
    melted = var_es.melt(id_vars="confianza", value_vars=["var_tve", "es_tve", "var_normal", "es_normal"], var_name="métrica", value_name="valor")
    labels = {"var_tve": "VaR TVE", "es_tve": "ES TVE", "var_normal": "VaR normal", "es_normal": "ES normal"}
    melted["métrica"] = melted["métrica"].map(labels)
    fig = px.bar(melted, x="confianza", y="valor", color="métrica", barmode="group", color_discrete_map={"VaR TVE": COLORS["navy"], "ES TVE": COLORS["blue"], "VaR normal": COLORS["gold"], "ES normal": COLORS["red"]})
    fig.update_xaxes(tickformat=".1%", title="Nivel de confianza")
    fig.update_yaxes(tickformat=".1%", title="Pérdida logarítmica")
    return style(fig, "VaR y Expected Shortfall: TVE frente a normal", 460)


def backtesting_figure(data: pd.DataFrame, var_value: float, confidence: float) -> go.Figure:
    validation = data[data["muestra"] == "Validación"].copy()
    breaches = validation[validation["perdida"] > var_value]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=validation["fecha"], y=validation["perdida"], mode="lines", line=dict(color=COLORS["blue"], width=1.2), name="Pérdida fuera de muestra"))
    fig.add_trace(go.Scatter(x=breaches["fecha"], y=breaches["perdida"], mode="markers", marker=dict(color=COLORS["red"], size=9), name="Excepción"))
    fig.add_hline(y=var_value, line_color=COLORS["red"], line_dash="dash", annotation_text=f"VaR TVE {confidence:.1%}")
    fig.update_yaxes(tickformat=".1%", title="Pérdida")
    return style(fig, "Backtesting fuera de muestra: pérdidas y excepciones", 460)


def simulation_figure(sim_data: pd.DataFrame, simulated_var: float, analytical_var: float) -> go.Figure:
    values = sim_data["perdida_simulada"].to_numpy()
    fig = go.Figure(go.Histogram(x=values, nbinsx=70, histnorm="probability density", marker_color=COLORS["cyan"], name="Pérdidas simuladas"))
    fig.add_vline(x=simulated_var, line_color=COLORS["red"], line_width=3, annotation_text="VaR simulado")
    fig.add_vline(x=analytical_var, line_color=COLORS["navy"], line_dash="dash", line_width=3, annotation_text="VaR analítico")
    fig.update_xaxes(tickformat=".1%", title="Pérdida simulada en la cola")
    fig.update_yaxes(title="Densidad")
    return style(fig, "Monte Carlo: distribución de pérdidas simuladas (sin fechas)", 460)
