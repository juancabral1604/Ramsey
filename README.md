# Aplicativo del modelo de Ramsey-Cass-Koopmans

Este repositorio contiene el motor matemático y numérico de un aplicativo interactivo para explorar el modelo de Ramsey-Cass-Koopmans.

## Componentes actuales

- Estado estacionario
- Jacobiano, autovalores y autovectores
- Aproximación lineal local de la senda estable
- Integración hacia atrás de la senda estable
- Problema de valores de frontera como control de robustez
- Simulación forward para un consumo inicial elegido
- Clasificación de trayectorias
- Comparación entre consumo bajo, consumo sobre la senda aproximada y consumo alto

## Estructura

- `ramsey_core.py`: motor matemático y numérico
- `ramsey_example.py`: ejecución de control con parámetros predeterminados
- `requirements.txt`: dependencias del proyecto

## Ejecución de control

```bash
python ramsey_example.py
```

## Advertencia metodológica

El estado estacionario presenta estabilidad de punto silla. En horizontes extensos, diferencias numéricas extremadamente pequeñas respecto de la senda estable pueden amplificarse por la presencia de un autovalor positivo. Los diagnósticos de convergencia se interpretan dentro del horizonte finito simulado.
