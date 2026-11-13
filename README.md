# IVT Nóvita

Caja de herramientas de ArcGIS Pro (Python/ArcPy) y scripts del Índice de Vulnerabilidad Territorial (IVT) de los territorios colectivos NARP de Nóvita (Chocó) frente a la minería formal de oro (títulos y solicitudes mineras).

## Contenido

- `toolbox/IVT_Novita_Toolbox.pyt`: cálculo del IVT.
- `scripts/IVT_robustez.py`: análisis de robustez.
- `scripts/IVT_validacion_hansen.py`: validación con pérdida de bosque.
- `results/`: tablas derivadas.

## Requisitos

ArcGIS Pro con la extensión Spatial Analyst y NumPy.

## Datos de entrada

No se incluyen. Son públicos: títulos y solicitudes mineras (ANM), consejos comunitarios, drenajes (IGAC), Global Forest Change v1.13 (Hansen et al., 2013; CC BY 4.0) y WorldPop R2025A.

## Uso

1. Cargar `toolbox/IVT_Novita_Toolbox.pyt` en ArcGIS Pro y ejecutar «IVT Nóvita: cálculo completo».
2. Para el IVT sin deforestación, ejecutar de nuevo con `variable_excluida = defor` y otro `nombre_gdb`.
3. Robustez y validación: editar las rutas del bloque de configuración de cada script y ejecutar `main()` en la ventana de Python de ArcGIS Pro.

## Licencia

MIT.
