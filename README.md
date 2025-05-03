# API de Predicción de Renovación Automática de Membresías de Supermercado

![Versión](https://img.shields.io/badge/versión-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-teal)
![Licencia](https://img.shields.io/badge/licencia-MIT-yellow)

Una API de aprendizaje automático para predecir la probabilidad de renovación automática de membresías de clientes de supermercado, basada en características del perfil de usuario y su comportamiento de compra.

## 📋 Contenido

- [Descripción General](#descripción-general)
- [Arquitectura](#arquitectura)
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
  - [API REST](#api-rest)
  - [Interfaz Streamlit](#interfaz-streamlit)
- [Modelo de ML](#modelo-de-ml)
- [Despliegue](#despliegue)
- [CI/CD](#cicd)
- [Monitoreo](#monitoreo)
- [Contribuir](#contribuir)

## 📝 Descripción General

Este proyecto proporciona una API para predecir si un cliente renovará automáticamente su membresía de supermercado, basándose en datos históricos de comportamiento y características demográficas. El sistema implementa un modelo de Machine Learning entrenado con datos de usuarios y está diseñado para integrarse fácilmente con aplicaciones existentes mediante una API REST.

## 🏗️ Arquitectura

El sistema está compuesto por los siguientes componentes:

1. **API REST**: Endpoint para realizar predicciones individuales y por lotes
2. **Modelo de ML**: Modelo de Random Forest entrenado para predecir renovaciones
3. **Interfaz Streamlit**: Dashboard para interactuar con el modelo
4. **Infraestructura**: Despliegue con Docker y Kubernetes
5. **CI/CD**: Automatización de pruebas, entrenamiento y despliegue

## ✨ Características

- Predicción de renovación con probabilidades
- Seguridad mediante API Key
- Procesamiento de solicitudes individuales y por lotes
- Interfaz visual para pruebas mediante Streamlit
- Reentrenamiento automático periódico
- Monitoreo de rendimiento del modelo
- Despliegue automatizado con Docker y Kubernetes
- Versión de modelos y registro de métricas

## 📦 Requisitos

```
pandas
scikit-learn
pyyaml
fastapi
uvicorn
joblib
pydantic
```

## 🚀 Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/usuario/grocery-membership-predictor.git
   cd grocery-membership-predictor
   ```

2. Crear entorno virtual (opcional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Configurar variables de entorno:
   ```bash
   export API_KEY="tu_clave_secreta"  # En Windows: set API_KEY=tu_clave_secreta
   ```

## 💻 Uso

### API REST

La API está construida con FastAPI y expone los siguientes endpoints:

#### Predicción Individual

```http
POST /predict
```

Cuerpo de la solicitud (JSON):

```json
{
  "gender": "male",
  "shared_account": true,
  "membership_tier": "silver",
  "membership_fee": 29.99,
  "push_notification_enabled": true,
  "have_app": true,
  "app_engagement_score": 65.5,
  "bought_store_brand": false,
  "promotion_participation_count": 3,
  "average_basket_size": 55.2,
  "use_count": 18,
  "reward_points_used": 200.0
}
```

Respuesta:

```json
{
  "auto_renew_prediction": 1,
  "probability_yes": 0.8763,
  "probability_no": 0.1237,
  "model_version": "v1.0.0",
  "prediction_id": "pred_f7g8h9",
  "prediction_timestamp": "2024-04-18T14:25:36Z"
}
```

#### Iniciar el Servidor

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Interfaz Streamlit

La aplicación incluye una interfaz web construida con Streamlit para facilitar las pruebas:

```bash
streamlit run app/streamlit/app.py
```

La interfaz estará disponible en http://localhost:8501.

## 🧠 Modelo de ML

El sistema utiliza un modelo de Random Forest Classifier entrenado con datos históricos de renovaciones de membresías:

- **Características de entrada**: Información demográfica, comportamiento de compra y uso de la app
- **Preprocesamiento**: Normalización de variables numéricas y codificación one-hot para categóricas
- **Pipeline**: Transformación + clasificación en un pipeline unificado
- **Métricas**: Precisión, recall, F1-score y AUC

Para reentrenar el modelo:

```bash
python app/models/train.py --data-path data/membership_groceries_userprofile.csv --output-path models/grocery_membership_model.joblib
```

## 🚢 Despliegue

### Docker Compose (Desarrollo)

```bash
docker-compose up -d
```

### Kubernetes (Producción)

El proyecto incluye configuraciones para desplegar en Kubernetes:

```bash
# Generar manifiestos
python utils/k8s_config_generator.py > k8s-deployment.yaml

# Aplicar configuración
kubectl apply -f k8s-deployment.yaml
```

## 🔄 CI/CD

El proyecto utiliza GitHub Actions para:

- **Integración Continua**: Pruebas automáticas en cada PR
- **Entrenamiento Periódico**: Reentrenamiento semanal del modelo
- **Despliegue Continuo**: Actualización automática en los entornos

### Flujos de trabajo

1. **CI**: Ejecuta pruebas en cada PR hacia main
2. **Reentrenamiento**: Actualiza el modelo semanalmente
3. **Despliegue**: Construye y despliega la imagen Docker

## 📊 Monitoreo

El sistema incluye monitoreo de:

- Rendimiento del modelo en producción
- Latencia de las predicciones  
- Distribución de datos de entrada
- Alertas de degradación de rendimiento

## 🤝 Contribuir

1. Haz un fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -m 'Añade nueva funcionalidad'`)
4. Sube la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## INTEGRANTES:
Junior Ramirez 
Jasser Palacios
Jason Barramtes
- 

