# Auditoría de fuentes

## Alcance

Se inspeccionaron una copia del documento Word, una copia del libro Excel y una copia de `complementarios.py`. Los originales no fueron modificados.

## Libro Excel

El libro contiene las hojas `Instrucciones`, `Datos`, `Descriptiva`, `Umbral`, `GPD`, `VaR_ES`, `RiskSim`, `Sim_RiskSimulator`, `Histogram`, `Sim_Datos` y `Backtesting`.

- `Datos` tiene 2.862 precios de cierre y 2.861 retornos/pérdidas diarios. La separación se calcula con fecha de corte 31 de diciembre de 2023.
- `Umbral` fija el percentil 90: u = 1,166776 %; n = 2.175; Nu = 218.
- `GPD` reporta ξ = 0,262445708, σ = 0,7331559 %, log-verosimilitud = 796,380449 y la conversión Pareto2 q = 3,8103119, b = 0,0279355.
- `VaR_ES` conserva VaR y ES TVE y normal para 95 %, 97,5 %, 99 % y 99,5 %.
- `Sim_Datos` contiene 10.000 pérdidas de cola simuladas. Se visualizan como simulaciones sin fechas, nunca como observaciones históricas.
- `Backtesting` contiene 16 pruebas de Kupiec: TVE y normal, dentro y fuera de muestra, para los cuatro niveles de confianza.

## Documento Word

El documento describe el mismo enfoque POT–GPD y coincide con el libro en el periodo, tamaños muestrales, umbral, parámetros, niveles de VaR/ES y resultados de Kupiec. También documenta que el VaR es fijo, no condicional, y que la simulación confirma el cálculo pero no valida por sí sola el modelo.

## Código complementario

El archivo original usa `google.colab.files.upload()` para recibir el Excel. La aplicación lo adapta a lectura local y conserva: sensibilidad a los umbrales 85, 90, 92,5 y 95; bootstrap con 2.000 réplicas y semilla 2026; y el escenario que excluye marzo y abril de 2020. El bootstrap se ejecuta solo con una acción explícita del usuario.

## Compatibilidad y límites

`RiskSim` almacena configuración interna del complemento en datos serializados, por lo que no se usa como fuente tabular. Los resultados verificables de la simulación están disponibles en `Sim_RiskSimulator` y `Sim_Datos`, que son las hojas consumidas por el tablero. No se detectaron discrepancias numéricas relevantes entre las tres fuentes.
