# proyecto_final_mlops
```markdown
# API de Predicción de Renovación Automática de Membresías de Supermercado

Este repositorio alberga un servicio basado en **FastAPI** y una serie de utilidades para **entrenar**, **versionar** y **desplegar** un modelo de **machine learning** que predice si un cliente renovará automáticamente su membresía de supermercado.

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Prerrequisitos](#prerrequisitos)
- [Instalación](#instalación)
- [Conjunto de Datos](#conjunto-de-datos)
- [Preprocesamiento y Limpieza](#preprocesamiento-y-limpieza)
- [Entrenamiento del Modelo (paso a paso)](#entrenamiento-del-modelo-paso-a-paso)
- [Registro y Versionado de Modelo](#registro-y-versionado-de-modelo)
- [Servicio API y Uso](#servicio-api-y-uso)
- [Configuración de Variables de Entorno](#configuración-de-variables-de-entorno)
- [Despliegue en Kubernetes](#despliegue-en-kubernetes)
- [Modo Desarrollo vs Producción](#modo-desarrollo-vs-producción)
- [Contribuciones](#contribuciones)

## Descripción del Proyecto

Este proyecto utiliza un modelo de **machine learning** cuyo objetivo es **predecir si un cliente renovará automáticamente su membresía de supermercado en su próxima facturación**. Para ello, el modelo calcula una probabilidad de renovación y emite una predicción binaria (0 = no renovará, 1 = renovará). Con esta información, las tiendas pueden anticipar el comportamiento de sus usuarios y diseñar campañas de retención más efectivas.

## Estructura del Proyecto

```
├── app
│   ├── api
│   │   └── main.py                # Punto de entrada de la API FastAPI
│   └── models
│       ├── schemas.py            # Definición de modelos Pydantic para validación de inputs/outputs
│       ├── train.py              # Script de entrenamiento del modelo
│       ├── versioning.py         # Lógica de registro/versionado de modelos
│       └── membership_model.joblib  # (Generado) modelo serializado
├── data
│   └── membership_groceries_userprofile.csv  # Dataset de perfiles de clientes
├── utils
│   └── k8s_config_generator.py  # Generador de manifiestos de Kubernetes
├── Dockerfile                   # Definición de imagen Docker multi-stage
├── requirements.txt             # Dependencias Python
└── README.md                    # Documentación del proyecto (este archivo)
```

## Prerrequisitos

- Python 3.10 o superior
- pip (`pip install --upgrade pip`)
- Docker (opcional, para contenerización)
- Acceso a la base de datos o CSV con perfiles de usuarios

## Instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/JunioX7X/trabajo_practica_4.git
   cd trabajo_practica_4
   ```
2. **Crear y activar un entorno virtual**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows
   ```
3. **Instalar dependencias**
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

## Conjunto de Datos

- El CSV de entrada debe ubicarse en `data/membership_groceries_userprofile.csv`.
- Columnas esperadas (entre otras):
  - `age` (edad)
  - `income` (ingresos anuales)
  - `shopping_frequency` (veces de compra por mes)
  - `avg_basket_value` (valor promedio del carrito)
  - `months_active` (meses de antigüedad)
  - `previous_renewals` (renovaciones previas)
  - `product_categories_purchased` (cantidad de categorías compradas)
  - `has_returned_items` (booleano: devoluciones previas)
  - `distance_to_store` (km al supermercado)
  - `membership_auto_renew` (target: 1 = renovó, 0 = no renovó)

## Preprocesamiento y Limpieza

1. **Carga de datos**: se lee el CSV con pandas.
2. **Tratamiento de valores faltantes**: imputación de medias para numéricos y moda para categóricos.
3. **Codificación**: una codificación one-hot para variables categóricas (p.ej., `has_returned_items`).
4. **Escalado**: normalización de características numéricas con `StandardScaler`.

## Entrenamiento del Modelo (paso a paso)

1. **Separación de características y target**:
   ```python
   X = df.drop("membership_auto_renew", axis=1)
   y = df["membership_auto_renew"]
   ```
2. **División entrenamiento/prueba**:
   - 80% datos para entrenamiento
   - 20% datos para validación
3. **Pipeline de preprocesamiento**:
   - `ColumnTransformer` con ramas para numéricos y categóricos
   - `StandardScaler` + `OneHotEncoder`
4. **Modelo**:
   - `RandomForestClassifier`
   - Hiperparámetros por defecto: 100 árboles, profundidad máxima 10
   - Se pueden modificar con flags al script `train.py`
5. **Entrenamiento**:
   ```bash
   python app/models/train.py \
     --data-path data/membership_groceries_userprofile.csv \
     --output-path app/models/membership_model.joblib
   ```
6. **Evaluación**:
   - Métricas: precisión, recall, F1-score, AUC-ROC
   - Se imprimen en consola al finalizar
7. **Serialización**:
   - El pipeline completo (preprocesamiento + modelo) se guarda con `joblib`

## Registro y Versionado de Modelo

El archivo `versioning.py` implementa un registro sencillo que:
1. Comprueba si ya existe un modelo con la versión indicada.
2. Mueve el modelo anterior a un subdirectorio `archive/` con timestamp.
3. Guarda la nueva versión en la ruta definida.

## Servicio API y Uso

1. **Arrancar el servidor**:
   ```bash
   export MODEL_PATH=app/models/membership_model.joblib
   export API_KEY=<tu_api_key_secreto>
   uvicorn app.api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Endpoint**: `POST /predict`
3. **Cabeceras**:
   - `X-API-Key: <tu_api_key>`
4. **Cuerpo (JSON)**: debe seguir el esquema `MembershipPredictorFeatures` en `schemas.py`.
5. **Respuesta**:
   ```json
   {
     "prediction": 1,
     "probability": { "renew": 0.87, "no_renew": 0.13 }
   }
   ```

### Ejemplo de petición

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 40,
    "income": 50000,
    "shopping_frequency": 10,
    "avg_basket_value": 80.5,
    "months_active": 12,
    "previous_renewals": 1,
    "product_categories_purchased": 5,
    "has_returned_items": false,
    "distance_to_store": 3.2
  }'
```

## Configuración de Variables de Entorno

- `MODEL_PATH`: ruta al archivo `.joblib` del modelo.
- `API_KEY`: clave secreta para autenticar peticiones.

## Despliegue en Kubernetes

Utiliza el generador de manifiestos en `utils/k8s_config_generator.py`:

```python
from utils.k8s_config_generator import generate_deployment_manifests
from app.models.schemas import ModelDeploymentConfig

config = ModelDeploymentConfig(
    model_path="/app/models/membership_model.joblib",
    version="v1.0.0",
    environment="production",
    replicas=3,
    resources={
        "requests": {"cpu": "500m", "memory": "512Mi"},
        "limits": {"cpu": "1", "memory": "1Gi"}
    },
    autoscaling_enabled=True,
    monitoring_enabled=True
)
manifests = generate_deployment_manifests(config)
print(manifests)
```

## Modo Desarrollo vs Producción

- **Desarrollo**: montado en vivo, logs detallados, recarga automática (Hot reload).
- **Producción**: construido en Docker multi-stage, sin código fuente en la imagen final, solo ejecuta Uvicorn.

## Contribuciones

¡Se valoran las mejoras! Para contribuir:
1. Abre un *issue* describiendo tu propuesta.
2. Realiza un *fork* y crea una rama nueva.
3. Envía un *pull request* con tu implementación.

---

**Autores**: Junios Ramirez, Jasser Palacios, Jason Barrantes
```

