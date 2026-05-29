# Estudio por simulación — Random Forest aplicado a California Housing
**Autores:** Bautista Torregiani, Juan Ignacio Castillo, Matías Fasolato, Santino Almirón Nanni

---

Este proyecto presenta un estudio de simulación sobre el comportamiento de un modelo **Random Forest Regressor** aplicado al dataset **California Housing**. El objetivo es analizar cómo distintos hiperparámetros del modelo afectan el error de predicción, el sesgo, la varianza y el costo computacional.

El trabajo se desarrolla en un notebook de Jupyter y sigue una estructura basada en etapas: formulación del problema, modelización conceptual, diseño de escenarios, implementación, experimentación y análisis de resultados.

---

## Objetivo del estudio

La pregunta principal del trabajo es:

> ¿Cómo afectan `max_depth` y `n_estimators` al MSE de test, al sesgo², a la varianza y al costo computacional de un Random Forest aplicado a la predicción del valor medio de viviendas en California?

En particular, se busca:

* evaluar cómo cambia el error de test al modificar la profundidad máxima de los árboles;
* analizar el efecto de la cantidad de árboles sobre el desempeño y el tiempo de entrenamiento;
* estimar la descomposición del error en sesgo², varianza y error irreducible;
* detectar posibles situaciones de subajuste o sobreajuste;
* identificar configuraciones con buen equilibrio entre desempeño, complejidad y costo computacional.

---

## Dataset

Se utiliza el dataset **California Housing**, incluido en `scikit-learn` mediante `fetch_california_housing`.

El dataset contiene:

* **20.640 observaciones**, correspondientes a distritos de California;
* **8 variables predictoras** relacionadas con características censales;
* una **variable respuesta**, que representa el valor medio de viviendas en cientos de miles de dólares.

Variables predictoras:

* `MedInc`
* `HouseAge`
* `AveRooms`
* `AveBedrms`
* `Population`
* `AveOccup`
* `Latitude`
* `Longitude`

Variable respuesta:

* `valor_vivienda`

El dataset no presenta valores faltantes.

---

## Modelo utilizado

El modelo utilizado es:

```python
RandomForestRegressor
```

Random Forest es un método de ensamble que combina muchos árboles de decisión. En problemas de regresión, la predicción final se obtiene promediando las predicciones de todos los árboles.

Este enfoque permite reducir la varianza respecto de un árbol individual, ya que los errores particulares de cada árbol tienden a compensarse al promediar sus predicciones.

---

## Diseño del estudio de simulación

Se evaluaron dos hiperparámetros principales:

| Hiperparámetro | Valores evaluados              |
| -------------- | ------------------------------ |
| `max_depth`    | 2, 3, 5, 7, 10, 15, 20, None   |
| `n_estimators` | 10, 25, 50, 100, 200, 300, 500 |

Para cada combinación se realizaron **30 réplicas**, utilizando distintas particiones train/test y distintas semillas aleatorias.

En total, se ejecutaron:

```text
8 × 7 × 30 = 1680 entrenamientos
```

---

## Métricas evaluadas

Las principales salidas del estudio fueron:

| Métrica             | Interpretación                                                       |
| ------------------- | -------------------------------------------------------------------- |
| `MSE_train`         | Error cuadrático medio en entrenamiento                              |
| `MSE_test`          | Error cuadrático medio en test                                       |
| `RMSE_test`         | Raíz del MSE de test, en la escala original de la variable respuesta |
| `R²_test`           | Proporción de variabilidad explicada por el modelo en test           |
| `tiempo_s`          | Tiempo de entrenamiento en segundos                                  |
| `sesgo²`            | Componente del error asociada a modelos demasiado simples            |
| `varianza`          | Variabilidad de las predicciones entre réplicas                      |
| `error irreducible` | Parte del error que no puede ser explicada por el modelo             |

El MSE se utiliza como métrica principal porque permite analizar la descomposición del error en sesgo², varianza y error irreducible.

---

## Estructura del notebook

El notebook principal es:

```text
estudio_simulacion_random_forest_california.ipynb
```

La estructura del trabajo es:

1. **Formulación del problema**
   Se define la pregunta de investigación, los objetivos y las hipótesis.

2. **Modelización conceptual**
   Se identifican la unidad de análisis, las variables, el modelo, los hiperparámetros y las salidas del estudio.

3. **Datos y configuración**
   Se carga el dataset California Housing y se revisan sus características principales.

4. **Diseño de escenarios**
   Se define la grilla de hiperparámetros y la cantidad de réplicas.

5. **Implementación**
   Se programan las funciones para entrenar modelos, calcular métricas y ejecutar réplicas.

6. **Experimentación**
   Se ejecuta la simulación principal y se estima la descomposición sesgo-varianza.

7. **Análisis de resultados y conclusiones**
   Se analizan gráficos, tablas y resultados finales.

---

## Resultados principales

Los resultados muestran que el hiperparámetro con mayor impacto sobre el desempeño fue `max_depth`.

Con profundidades bajas, como `max_depth = 2` o `max_depth = 3`, el modelo presenta valores altos de MSE de test. Esto indica subajuste, ya que los árboles son demasiado simples para capturar la relación entre las variables predictoras y el valor de la vivienda.

Al aumentar `max_depth`, el MSE de test disminuye de forma marcada. Sin embargo, a partir de profundidades altas, como `max_depth = 15`, `20` o `None`, las mejoras se vuelven pequeñas. Por este motivo, una profundidad intermedia como `max_depth = 10` puede interpretarse como un punto de equilibrio razonable entre desempeño y complejidad.

Por otro lado, `n_estimators` también influye en el modelo, pero de forma más limitada. Aumentar la cantidad de árboles reduce el MSE de test principalmente al comienzo, pero luego aparecen rendimientos decrecientes. Además, el tiempo de entrenamiento aumenta de manera clara a medida que se agregan más árboles.

---

## Conclusión general

El estudio muestra que Random Forest es un modelo adecuado para predecir el valor medio de viviendas en el dataset California Housing.

El principal factor que afecta el error de predicción es la profundidad máxima de los árboles. Profundidades demasiado bajas generan subajuste, mientras que profundidades intermedias o altas reducen considerablemente el error de test.

La cantidad de árboles mejora la estabilidad del modelo, pero después de cierto punto las mejoras son marginales en comparación con el aumento del tiempo de entrenamiento.

En conjunto, los resultados muestran que el mejor modelo no necesariamente es el más complejo. Una configuración intermedia puede lograr un buen equilibrio entre error de predicción, estabilidad y costo computacional.

---

## Cómo ejecutar el proyecto

1. Clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

2. Crear y activar un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecutar el notebook:

```bash
jupyter notebook estudio_simulacion_random_forest_california.ipynb
```

También puede abrirse directamente desde Visual Studio Code.

---

## Dependencias principales

El proyecto utiliza las siguientes librerías:

* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `jupyter`

---

## Archivos principales

```text
.
├── estudio_simulacion_random_forest_california.ipynb
├── README.md
├── requirements.txt
└── outputs/
```

