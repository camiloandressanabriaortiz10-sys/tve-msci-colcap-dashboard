# MSCI COLCAP | Extreme Risk Analytics

Aplicación local en Streamlit para explorar la Teoría de Valor Extremo (TVE) aplicada al MSCI COLCAP. Presenta los resultados originales del modelo POT–GPD del proyecto y no escribe sobre los archivos fuente.

## Contenido

- `app.py`: interfaz de Streamlit.
- `modules/`: lectura del libro, cálculos complementarios y visualizaciones.
- `data/Modulo2_TVE_COLCAP_RiskSimulator_FINAL_copia.xlsx`: copia de trabajo del libro original.
- `complementarios_original_copia.py`: copia preservada del código de Colab.
- `tests/test_audit.py`: controles reproducibles de integridad.

## Ejecutar en Windows

1. Abra PowerShell dentro de esta carpeta.
2. Cree un entorno virtual: `py -m venv .venv`
3. Actívelo: `.\.venv\Scripts\Activate.ps1`
4. Instale dependencias: `pip install -r requirements.txt`
5. Ejecute: `streamlit run app.py`

Se abrirá el navegador con la aplicación local. Para detenerla, use `Ctrl+C` en PowerShell.

## Fuente y trazabilidad

La aplicación carga por defecto la copia del Excel incluida en `data/` y usa como resultados de referencia sus hojas `Datos`, `Umbral`, `GPD`, `VaR_ES`, `Sim_RiskSimulator`, `Sim_Datos` y `Backtesting`. Lee únicamente valores calculados y no altera Solver, Risk Simulator ni las fórmulas del libro.

El selector de archivo de la barra lateral acepta otra copia compatible. Antes de cargarla, verifica las hojas requeridas. El libro que se seleccione se usa únicamente en modo lectura.

## Módulos

1. Resumen general: controles, parámetros y resultados base.
2. Pérdidas en el tiempo: pérdidas diarias, separación estimación/validación, umbral y VaR fijo.
3. TVE y GPD: excedencias, distribución GPD y diagnóstico de umbral.
4. VaR y ES: comparación TVE versus normal para 95 %, 97,5 %, 99 % y 99,5 %.
5. Backtesting: prueba de Kupiec, excepciones y fechas de violación fuera de muestra.
6. Monte Carlo: simulaciones de cola, sin presentar observaciones simuladas como datos históricos.
7. Complementarios: sensibilidad y exclusión de marzo-abril de 2020, además de bootstrap bajo demanda.

## Verificación

Con el entorno activado, ejecute:

`python tests/test_audit.py`

El control verifica 2.861 pérdidas, fechas ordenadas y únicas, 2.175 observaciones de estimación, ξ, σ, el VaR TVE al 99 %, las 10.000 simulaciones y los 16 resultados de Kupiec.

## Limitaciones

El VaR mostrado es fijo, pues es el que reporta el modelo POT–GPD del proyecto. No se implementa GARCH ni un VaR condicional. El bootstrap conserva el enfoque de `complementarios.py` y se ejecuta expresamente por su costo computacional.
