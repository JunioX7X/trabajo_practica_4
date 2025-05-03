# app/models/train.py
import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(data_path: str, output_path: str, hyperparams: dict = None):
    """
    Entrena un modelo de predicción de membresías de supermercado.

    Args:
        data_path: Ruta al CSV de perfiles de usuario
        output_path: Ruta donde se guardará el modelo serializado
        hyperparams: Parámetros opcionales para configurar el modelo

    Returns:
        dict: Métricas de rendimiento y metadatos del modelo
    """

    logger.info(f"Iniciando entrenamiento con datos desde: {data_path}")

    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Cargar dataset
    df = pd.read_csv(data_path)
    logger.info(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} características")

    TARGET_COL = "membership_auto_renew"

    # Preprocesamiento
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # Codificación de variables categóricas
    X = pd.get_dummies(X)

    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Split de datos completado: {X_train.shape[0]} ejemplos de entrenamiento")

    # Configuración de pipeline con escalado y modelo
    hyperparams = hyperparams or {'n_estimators': 100, 'max_depth': 10}

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**hyperparams))
    ])

    # Entrenamiento
    logger.info("Iniciando entrenamiento del modelo...")
    pipeline.fit(X_train, y_train)

    # Evaluación
    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    logger.info(f"Entrenamiento completado - Accuracy entrenamiento: {train_accuracy:.4f}, "
                f"Accuracy prueba: {test_accuracy:.4f}")

    # Serialización
    logger.info(f"Guardando modelo en: {output_path}")
    joblib.dump(pipeline, output_path)

    # Calcular importancia de características si el modelo lo soporta
    feature_importance = {}
    if hasattr(pipeline['classifier'], 'feature_importances_'):
        feature_importance = dict(zip(X.columns, pipeline['classifier'].feature_importances_))

    # Metadatos del modelo
    model_info = {
        "metrics": {
            "accuracy_train": float(train_accuracy),
            "accuracy_test": float(test_accuracy)
        },
        "hyperparameters": hyperparams,
        "feature_importance": feature_importance,
        "data_source": data_path,
        "n_features": X.shape[1],
        "n_samples_train": X_train.shape[0],
        "feature_names": list(X.columns)
    }

    logger.info("Proceso de entrenamiento finalizado exitosamente")
    # Guardar columnas utilizadas para entrenamiento
    columns_path = output_path.replace(".joblib", "_columns.json")
    with open(columns_path, "w") as f:
        json.dump(model_info["feature_names"], f)

    return model_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrenamiento de modelo para predicción de membresías")
    parser.add_argument("--data-path", type=str, required=True, help="Ruta al archivo CSV de datos")
    parser.add_argument("--output-path", type=str, required=True, help="Ruta donde guardar el modelo entrenado")
    args = parser.parse_args()

    train_model(args.data_path, args.output_path)
    # Guardar columnas usadas para entrenar el modelo


