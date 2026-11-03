from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from modules.data_loader import audit_checks, default_workbook, load_model_data, validate_workbook
from modules.risk_metrics import bootstrap_95, losses_to_cop, sensitivity_and_exclusion, tve_risk
from modules.visualizations import (
    COLORS, backtesting_figure, gpd_figure, loss_timeline, risk_comparison,
    simulation_figure, threshold_figure,
)


st.set_page_config(page_title="MSCI COLCAP | Extreme Risk Analytics", page_icon="📈", layout="wide", initial_sidebar_state="expanded")


def inject_css() -> None:
    css = (Path(__file__).parent / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Leyendo resultados verificados del Excel…")
def get_model(path: str, modified: float) -> dict:
    del modified
    return load_model_data(path)


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}".replace(".", ",")


def cop(value: float, millions: bool = False) -> str:
    if millions:
        return f"COP {value / 1_000_000:,.1f} mill.".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"COP {value:,.0f}".replace(",", ".")


def pretty_table(df: pd.DataFrame, percent_cols: list[str] | None = None, money_cols: list[str] | None = None) -> pd.io.formats.style.Styler:
    percent_cols, money_cols = percent_cols or [], money_cols or []
    formatters = {column: (lambda x: pct(x)) for column in percent_cols if column in df.columns}
    formatters.update({column: (lambda x: cop(x)) for column in money_cols if column in df.columns})
    return df.style.format(formatters, na_rep="—")


def render_title(model: dict) -> None:
    st.markdown("<div class='hero'><div><div class='eyebrow'>RIESGO DE MERCADO · TVE</div><h1>MSCI COLCAP | Extreme Risk Analytics</h1><p>Exploración interactiva del modelo POT–GPD para pérdidas extremas del índice colombiano.</p></div><div class='hero-badge'>Modelo fijo<br><b>POT–GPD</b></div></div>", unsafe_allow_html=True)
    st.caption(f"Fuente y trazabilidad: {model['source']} La aplicación lee el libro y no modifica sus parámetros ni resultados.")


def select_confidence(model: dict, key: str) -> float:
    values = model["var_es"]["confianza"].tolist()
    selected = st.select_slider("Nivel de confianza", options=values, value=0.99, key=key, format_func=lambda v: pct(v, 1))
    return float(selected)


def overview(model: dict) -> None:
    p, data, risk = model["params"], model["data"], model["var_es"]
    chosen = risk.loc[np.isclose(risk["confianza"], 0.99)].iloc[0]
    st.subheader("Resumen del modelo")
    st.write("El modelo convierte retornos logarítmicos diarios en pérdidas, modela los excesos sobre el percentil 90 con una GPD y usa el resultado para estimar VaR y Expected Shortfall. No es un VaR condicional ni un modelo GARCH.")
    cols = st.columns(5)
    cols[0].metric("Pérdidas diarias", f"{len(data):,}")
    cols[1].metric("Muestra de estimación", f"{p['n']:,}", "2015–2023")
    cols[2].metric("Umbral POT", pct(p["umbral"]))
    cols[3].metric("Forma ξ", f"{p['xi']:.4f}")
    cols[4].metric("Escala σ", pct(p["sigma"]))
    cols2 = st.columns(3)
    cols2[0].metric("VaR TVE 99 %", pct(chosen["var_tve"]), cop(chosen["var_tve_cop"], True))
    cols2[1].metric("ES TVE 99 %", pct(chosen["es_tve"]), cop(chosen["es_tve_cop"], True))
    cols2[2].metric("Excesos observados", f"{p['nu']:,}", f"{p['nu']/p['n']:.2%} de estimación")

    left, right = st.columns((1.2, 1))
    with left:
        st.plotly_chart(loss_timeline(data, p["umbral"], chosen["var_tve"], 0.99), use_container_width=True)
    with right:
        st.subheader("Controles de integridad")
        for name, passed, note in audit_checks(model):
            st.markdown(f"<div class='check {'pass' if passed else 'fail'}'><b>{'✓' if passed else '!' } {name}</b><br><span>{note}</span></div>", unsafe_allow_html=True)
        st.caption("Los controles se ejecutan sobre la copia del Excel que usa la aplicación.")


def temporal_analysis(model: dict) -> None:
    st.subheader("Análisis temporal de pérdidas")
    st.write("Cada punto representa un día de negociación. Las líneas son referencias fijas del modelo POT–GPD, no pronósticos diarios variables.")
    confidence = select_confidence(model, "timeline_confidence")
    risk = model["var_es"].loc[np.isclose(model["var_es"]["confianza"], confidence)].iloc[0]
    dates = model["data"]["fecha"]
    start, end = st.slider("Periodo visible", min_value=dates.min().date(), max_value=dates.max().date(), value=(dates.min().date(), dates.max().date()), key="timeline_dates")
    shown = model["data"].loc[model["data"]["fecha"].between(pd.Timestamp(start), pd.Timestamp(end))]
    st.plotly_chart(loss_timeline(shown, model["params"]["umbral"], risk["var_tve"], confidence), use_container_width=True)
    exceptions = shown.loc[shown["perdida"] > risk["var_tve"], ["fecha", "retorno", "perdida", "muestra"]]
    st.metric("Excepciones en el rango visible", len(exceptions), f"Pérdida > VaR TVE {pct(confidence, 1)}")
    if not exceptions.empty:
        st.dataframe(pretty_table(exceptions.rename(columns={"fecha": "Fecha", "retorno": "Retorno", "perdida": "Pérdida", "muestra": "Periodo"}), ["Retorno", "Pérdida"]), use_container_width=True, hide_index=True)


def gpd_analysis(model: dict) -> None:
    p, data = model["params"], model["data"]
    st.subheader("Teoría de Valor Extremo y distribución GPD")
    left, right = st.columns((1.35, 1))
    with left:
        st.plotly_chart(gpd_figure(data.loc[data["muestra"] == "Estimación", "perdida"].to_numpy(), p["umbral"], p["xi"], p["sigma"]), use_container_width=True)
    with right:
        st.markdown("#### Parámetros estimados")
        st.metric("ξ — forma", f"{p['xi']:.4f}")
        st.write("ξ positivo indica una cola pesada: las pérdidas extremas se atenúan más lentamente que bajo normalidad.")
        st.metric("σ — escala", pct(p["sigma"]))
        st.write("σ mide la dispersión de los excesos sobre el umbral. No es el VaR ni el umbral.")
        st.metric("u — umbral", pct(p["umbral"]))
        st.write(f"Hay {p['nu']} excesos que superan u dentro de la muestra de estimación.")
    st.plotly_chart(threshold_figure(model["threshold_table"], 0.90), use_container_width=True)
    st.caption("El libro selecciona el percentil 90 como equilibrio entre contar suficientes excesos y conservar un comportamiento estable de la cola.")


def var_es_analysis(model: dict) -> None:
    st.subheader("VaR y Expected Shortfall")
    st.write("El VaR marca el umbral de pérdida para cada confianza. El ES mide la pérdida promedio una vez que ese VaR fue superado.")
    risk = model["var_es"].copy()
    st.plotly_chart(risk_comparison(risk), use_container_width=True)
    display = risk.rename(columns={"confianza":"Confianza", "var_tve":"VaR TVE", "es_tve":"ES TVE", "var_normal":"VaR normal", "es_normal":"ES normal", "ratio_var":"TVE/normal", "var_tve_cop":"VaR TVE COP", "es_tve_cop":"ES TVE COP"})
    st.dataframe(pretty_table(display[["Confianza", "VaR TVE", "ES TVE", "VaR normal", "ES normal", "TVE/normal", "VaR TVE COP", "ES TVE COP"]], ["Confianza", "VaR TVE", "ES TVE", "VaR normal", "ES normal"], ["VaR TVE COP", "ES TVE COP"]), use_container_width=True, hide_index=True)
    st.caption("Equivalentes monetarios para el portafolio de referencia de COP 1.000 millones y la transformación V·(1−e⁻ᴸ) definida en el libro.")


def backtesting(model: dict) -> None:
    st.subheader("Backtesting")
    st.write("Una excepción ocurre solo cuando la pérdida diaria supera el VaR. No equivale a superar el umbral POT.")
    confidence = select_confidence(model, "backtesting_confidence")
    results = model["backtesting"]
    selected = results.loc[np.isclose(results["confianza"], confidence)].copy()
    tve_out = selected[(selected["modelo"] == "TVE") & selected["muestra"].str.contains("Validación")].iloc[0]
    st.plotly_chart(backtesting_figure(model["data"], tve_out["var"], confidence), use_container_width=True)
    cards = st.columns(4)
    cards[0].metric("Días de validación", f"{int(tve_out['dias']):,}")
    cards[1].metric("Excepciones TVE", int(tve_out["violaciones"]))
    cards[2].metric("Tasa observada", pct(tve_out["tasa"]))
    cards[3].metric("Kupiec LR", f"{tve_out['lr_kupiec']:.3f}", tve_out["decision"])
    display = selected.rename(columns={"muestra":"Muestra", "modelo":"Modelo", "confianza":"Confianza", "var":"VaR", "dias":"T días", "violaciones":"Excepciones", "esperadas":"Esperadas", "tasa":"Tasa", "lr_kupiec":"LR Kupiec", "p_valor":"p-valor", "decision":"Decisión"})
    st.dataframe(pretty_table(display, ["Confianza", "VaR", "Tasa", "p-valor"]), use_container_width=True, hide_index=True)
    dates = model["data"].query("muestra == 'Validación' and perdida > @tve_out.var")[['fecha', 'perdida']]
    st.write("Fechas de excepciones TVE fuera de muestra:")
    st.dataframe(pretty_table(dates.rename(columns={"fecha":"Fecha", "perdida":"Pérdida"}), ["Pérdida"]), hide_index=True)


def simulations(model: dict) -> None:
    st.subheader("Simulación Monte Carlo")
    st.write("Son 10.000 pérdidas generadas en la cola GPD. No son una serie histórica y no se asignan fechas reales.")
    confidence = select_confidence(model, "simulation_confidence")
    row = model["simulation_summary"].loc[np.isclose(model["simulation_summary"]["confianza"], confidence)].iloc[0]
    st.plotly_chart(simulation_figure(model["sim_data"], row["var_sim"], row["var_analitico"]), use_container_width=True)
    cards = st.columns(4)
    cards[0].metric("Ensayos", f"{len(model['sim_data']):,}")
    cards[1].metric("VaR simulado", pct(row["var_sim"]))
    cards[2].metric("VaR analítico", pct(row["var_analitico"]))
    cards[3].metric("Diferencia VaR", pct(row["dif_var"]))
    display = model["simulation_summary"].rename(columns={"confianza":"Confianza", "p_cola":"Percentil cola p*", "var_sim":"VaR simulado", "es_sim":"ES simulado", "var_analitico":"VaR analítico", "es_analitico":"ES analítico", "dif_var":"Diferencia VaR", "dif_es":"Diferencia ES", "rango_var":"Rango VaR (95%)", "rango_es":"Rango ES (95%)"})
    st.dataframe(pretty_table(display[["Confianza", "Percentil cola p*", "VaR simulado", "ES simulado", "VaR analítico", "ES analítico", "Diferencia VaR", "Diferencia ES", "Rango VaR (95%)", "Rango ES (95%)"]], ["Confianza", "Percentil cola p*", "VaR simulado", "ES simulado", "VaR analítico", "ES analítico", "Diferencia VaR", "Diferencia ES"]), use_container_width=True, hide_index=True)


def complementary(model: dict) -> None:
    st.subheader("Análisis complementarios")
    st.write("Estos cálculos adaptan `complementarios.py` para ejecución local. Conservan la estimación GPD por Nelder–Mead y no modifican el Excel fuente.")
    losses = model["data"].loc[model["data"]["muestra"] == "Estimación", "perdida"].to_numpy()
    p = model["params"]
    sensitivity = pd.DataFrame(sensitivity_and_exclusion(losses, p["portafolio"]))
    excluded = model["data"].loc[(model["data"]["muestra"] == "Estimación") & ~((model["data"]["fecha"].dt.year == 2020) & (model["data"]["fecha"].dt.month.isin([3, 4]))), "perdida"].to_numpy()
    u = np.quantile(excluded, .90)
    excesses = excluded[excluded > u] - u
    from modules.risk_metrics import fit_gpd
    xi, sigma = fit_gpd(excesses)
    var, es = tve_risk(xi, sigma, u, len(excluded), len(excesses))
    sensitivity = pd.concat([sensitivity, pd.DataFrame([{"escenario": "Sin marzo y abril de 2020", "xi": xi, "var_99": var, "es_99": es, "var_cop": float(losses_to_cop(var, p["portafolio"])), "es_cop": float(losses_to_cop(es, p["portafolio"]))}])], ignore_index=True)
    display = sensitivity.rename(columns={"escenario":"Escenario", "xi":"ξ", "var_99":"VaR 99 %", "es_99":"ES 99 %", "var_cop":"VaR COP", "es_cop":"ES COP"})
    st.dataframe(pretty_table(display, ["VaR 99 %", "ES 99 %"], ["VaR COP", "ES COP"]), use_container_width=True, hide_index=True)
    st.caption("El escenario sin marzo y abril de 2020 no es una recomendación de exclusión; ilustra cuánto dependen los resultados de la muestra de crisis.")
    st.divider()
    st.markdown("#### Bootstrap al 95 %")
    st.write("El remuestreo usa 2.000 réplicas y semilla 2026. Puede tardar; se ejecuta únicamente al solicitarlo y luego queda almacenado durante esta sesión.")
    if st.button("Ejecutar bootstrap de 2.000 réplicas", type="primary"):
        with st.spinner("Ejecutando las 2.000 réplicas…"):
            st.session_state["bootstrap"] = bootstrap_95(losses)
    result = st.session_state.get("bootstrap")
    if result:
        intervals = result["intervalos"]
        table = pd.DataFrame({"Métrica": ["ξ", "VaR 99 %", "ES 99 %"], "Límite inferior": intervals[0], "Límite superior": intervals[1]})
        st.dataframe(pretty_table(table, ["Límite inferior", "Límite superior"]), hide_index=True, use_container_width=True)
        st.caption(f"Resultado reproducible con semilla {result['seed']} y {result['replicas']:,} réplicas.")


def main() -> None:
    inject_css()
    st.sidebar.markdown("## Navegación")
    default = default_workbook()
    upload = st.sidebar.file_uploader("Usar otra copia del Excel", type=["xlsx"], help="La carga se procesa como lectura. El libro de origen no se modifica.")
    path = default
    temp_path = None
    if upload is not None:
        temp_dir = Path(tempfile.gettempdir()) / "tve_colcap_dashboard"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / upload.name
        temp_path.write_bytes(upload.getbuffer())
        path = temp_path
    valid, message = validate_workbook(path)
    if not valid:
        st.error(message)
        st.stop()
    model = get_model(str(path), path.stat().st_mtime)
    st.sidebar.success("Libro validado · lectura solamente")
    page = st.sidebar.radio("Sección", ["Resumen general", "Pérdidas en el tiempo", "TVE y GPD", "VaR y ES", "Backtesting", "Monte Carlo", "Complementarios"])
    st.sidebar.divider()
    st.sidebar.caption("Proyecto académico · Ingeniería Financiera UNAB")
    render_title(model)
    pages = {
        "Resumen general": overview, "Pérdidas en el tiempo": temporal_analysis, "TVE y GPD": gpd_analysis,
        "VaR y ES": var_es_analysis, "Backtesting": backtesting, "Monte Carlo": simulations,
        "Complementarios": complementary,
    }
    pages[page](model)


if __name__ == "__main__":
    main()
