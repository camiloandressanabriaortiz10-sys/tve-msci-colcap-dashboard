from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd


REQUIRED_SHEETS = {
    "Datos", "Descriptiva", "Umbral", "GPD", "VaR_ES",
    "Sim_RiskSimulator", "Sim_Datos", "Backtesting",
}


def default_workbook() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "Modulo2_TVE_COLCAP_RiskSimulator_FINAL_copia.xlsx"


def validate_workbook(path: str | Path) -> tuple[bool, str]:
    """Comprueba que un libro contenga las hojas mínimas, sin modificarlo."""
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        missing = REQUIRED_SHEETS.difference(wb.sheetnames)
        if missing:
            return False, f"Faltan hojas requeridas: {', '.join(sorted(missing))}."
        return True, "Libro validado."
    except Exception as exc:  # La interfaz presenta el mensaje de forma controlada.
        return False, f"No fue posible leer el libro: {exc}"


def _records(ws, start_row: int, headers: list[str], start_col: int = 1) -> pd.DataFrame:
    values = []
    for row in ws.iter_rows(min_row=start_row, min_col=start_col, max_col=start_col + len(headers) - 1, values_only=True):
        if any(value is not None for value in row):
            values.append(row)
    return pd.DataFrame(values, columns=headers)


def _scalar(ws, cell: str) -> Any:
    return ws[cell].value


def load_model_data(path: str | Path) -> dict[str, Any]:
    """Lee resultados almacenados en Excel; no recalcula ni escribe en el libro."""
    path = Path(path)
    valid, message = validate_workbook(path)
    if not valid:
        raise ValueError(message)

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Datos"]
    data = _records(ws, 3, ["fecha", "cierre", "retorno", "perdida", "muestra"])
    data["fecha"] = pd.to_datetime(data["fecha"])
    for col in ["cierre", "retorno", "perdida"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["fecha", "perdida"]).sort_values("fecha").reset_index(drop=True)

    uw = wb["Umbral"]
    threshold_table = _records(uw, 11, ["percentil", "umbral", "nu", "exceso_medio"], 2)
    for col in ["percentil", "umbral", "nu", "exceso_medio"]:
        threshold_table[col] = pd.to_numeric(threshold_table[col], errors="coerce")

    gpd = wb["GPD"]
    params = {
        "umbral": float(_scalar(gpd, "C3")),
        "xi": float(_scalar(gpd, "C4")),
        "sigma": float(_scalar(gpd, "C5")),
        "nu": int(_scalar(gpd, "C6")),
        "log_verosimilitud": float(_scalar(gpd, "C7")),
        "q_pareto2": float(_scalar(gpd, "C8")),
        "b_pareto2": float(_scalar(gpd, "C9")),
        "n": int(_scalar(wb["VaR_ES"], "C6")),
        "portafolio": float(_scalar(wb["VaR_ES"], "C10")),
    }

    var_ws = wb["VaR_ES"]
    var_es = _records(
        var_ws, 13,
        ["confianza", "var_tve", "es_tve", "var_normal", "es_normal", "ratio_var", "var_tve_cop", "es_tve_cop", "var_normal_cop", "es_normal_cop"],
        2,
    )
    for col in var_es.columns:
        var_es[col] = pd.to_numeric(var_es[col], errors="coerce")

    sim_ws = wb["Sim_RiskSimulator"]
    simulation_summary = _records(
        sim_ws, 15,
        ["confianza", "p_cola", "var_sim", "es_sim", "var_analitico", "es_analitico", "dif_var", "dif_es", "control_var", "control_es", "rango_var", "rango_es"],
        2,
    )
    for col in simulation_summary.columns[:10]:
        simulation_summary[col] = pd.to_numeric(simulation_summary[col], errors="coerce")

    sim_data = _records(wb["Sim_Datos"], 4, ["perdida_simulada"])
    sim_data["perdida_simulada"] = pd.to_numeric(sim_data["perdida_simulada"], errors="coerce")
    sim_data = sim_data.dropna()

    bt_ws = wb["Backtesting"]
    backtesting = _records(
        bt_ws, 6,
        ["muestra", "modelo", "confianza", "var", "dias", "violaciones", "esperadas", "tasa", "lr_kupiec", "p_valor", "decision"],
        2,
    )
    # Las filas 6:21 contienen el resumen de Kupiec. Las filas siguientes son el detalle diario.
    backtesting = backtesting.iloc[:16].copy()
    for col in ["confianza", "var", "dias", "violaciones", "esperadas", "tasa", "lr_kupiec", "p_valor"]:
        backtesting[col] = pd.to_numeric(backtesting[col], errors="coerce")
    backtesting["violaciones"] = backtesting["violaciones"].astype(int)
    backtesting["dias"] = backtesting["dias"].astype(int)

    desc = _records(wb["Descriptiva"], 4, ["estadistico", "estimacion", "validacion", "total"])

    return {
        "path": str(path), "data": data, "params": params, "threshold_table": threshold_table,
        "var_es": var_es, "sim_data": sim_data, "simulation_summary": simulation_summary,
        "backtesting": backtesting, "descriptive": desc,
        "source": "Banco de la República (2026), Índice COLCAP, serie 6; cálculos del libro suministrado.",
    }


def audit_checks(model: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Controles de integridad que comparan los datos leídos con los resultados fuente."""
    data, p, risk = model["data"], model["params"], model["var_es"]
    checks = [
        ("Observaciones", len(data) == 2861, f"{len(data):,} pérdidas leídas."),
        ("Fechas ordenadas", data["fecha"].is_monotonic_increasing, "Serie ordenada de menor a mayor fecha."),
        ("Fechas únicas", not data["fecha"].duplicated().any(), "No se detectaron fechas duplicadas."),
        ("Separación de muestras", int((data["muestra"] == "Estimación").sum()) == p["n"], f"{p['n']:,} observaciones de estimación."),
        ("Parámetros GPD", np.isclose(p["xi"], 0.262445708) and np.isclose(p["sigma"], 0.007331559), f"ξ={p['xi']:.9f}; σ={p['sigma']:.9f}."),
        ("VaR 99 %", np.isclose(float(risk.loc[risk["confianza"] == 0.99, "var_tve"].iloc[0]), 0.03488443011631677), "Coincide con el resultado fuente."),
    ]
    return checks
