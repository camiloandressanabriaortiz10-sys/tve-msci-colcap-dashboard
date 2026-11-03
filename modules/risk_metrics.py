from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def gpd_pdf(y: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    z = 1 + xi * y / sigma
    output = np.zeros_like(y)
    valid = z > 0
    if abs(xi) < 1e-7:
        output[valid] = np.exp(-y[valid] / sigma) / sigma
    else:
        output[valid] = (1 / sigma) * z[valid] ** (-1 / xi - 1)
    return output


def gpd_cdf(y: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    z = 1 + xi * y / sigma
    if abs(xi) < 1e-7:
        return 1 - np.exp(-y / sigma)
    return np.where(z > 0, 1 - z ** (-1 / xi), np.nan)


def losses_to_cop(losses: np.ndarray | float, portfolio: float) -> np.ndarray | float:
    return portfolio * (1 - np.exp(-np.asarray(losses)))


def _nll(par: np.ndarray, excesses: np.ndarray) -> float:
    xi, sigma = par
    if sigma <= 1e-9 or np.any(1 + xi * excesses / sigma <= 0):
        return 1e9
    z = 1 + xi * excesses / sigma
    if abs(xi) < 1e-7:
        return len(excesses) * np.log(sigma) + np.sum(excesses / sigma)
    return len(excesses) * np.log(sigma) + (1 + 1 / xi) * np.log(z).sum()


def fit_gpd(excesses: np.ndarray, starts=((0.2, 0.007), (0.05, 0.009), (0.3, 0.006))) -> tuple[float, float]:
    results = [minimize(_nll, start, args=(excesses,), method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-10}) for start in starts]
    best = min(results, key=lambda result: result.fun)
    return float(best.x[0]), float(best.x[1])


def tve_risk(xi: float, sigma: float, threshold: float, n: int, nu: int, confidence: float = 0.99) -> tuple[float, float]:
    var = threshold + sigma / xi * (((n / nu) * (1 - confidence)) ** (-xi) - 1)
    es = (var + sigma - xi * threshold) / (1 - xi)
    return float(var), float(es)


def sensitivity_and_exclusion(losses: np.ndarray, portfolio: float) -> list[dict]:
    """Replica las dos partes no-bootstrap de complementarios.py."""
    outputs = []
    for q in (0.85, 0.90, 0.925, 0.95):
        u = float(np.quantile(losses, q))
        y = losses[losses > u] - u
        xi, sigma = fit_gpd(y)
        var, es = tve_risk(xi, sigma, u, len(losses), len(y))
        outputs.append({"escenario": f"Umbral p{q * 100:g}", "xi": xi, "var_99": var, "es_99": es, "var_cop": float(losses_to_cop(var, portfolio)), "es_cop": float(losses_to_cop(es, portfolio))})
    return outputs


def bootstrap_95(losses: np.ndarray, replicas: int = 2000, seed: int = 2026) -> dict:
    """Replica el bootstrap del archivo original y se invoca solo bajo demanda."""
    u = float(np.quantile(losses, 0.90))
    y = losses[losses > u] - u
    xi, sigma = fit_gpd(y)
    rng = np.random.default_rng(seed)
    simulation = []
    for _ in range(replicas):
        sample = rng.choice(y, len(y), replace=True)
        xb, sb = fit_gpd(sample, starts=((xi, sigma), (0.05, 0.009)))
        vb, eb = tve_risk(xb, sb, u, len(losses), len(y))
        simulation.append((xb, vb, eb))
    values = np.asarray(simulation)
    return {"replicas": replicas, "seed": seed, "intervalos": np.percentile(values, [2.5, 97.5], axis=0), "muestras": values}
