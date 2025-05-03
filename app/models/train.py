# app/models/train.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, output_path: str, hyperparams: dict = None):

    logger.info(f"Iniciando entrenamiento con datos desde: {data_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(data_path)
    logger.info(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} características")
    print("Columnas disponibles en el CSV:", df.columns.tolist())

    TARGET_COL = "membership_auto_renew"
    y = df[TARGET_COL]

    # Solo entrenamos con las columnas esperadas por la API
    input_features = [
        "gender", "shared_account", "membership_tier", "membership_fee",
        "push_notification_enabled", "have_app", "app_engagement_score",
        "bought_store_brand", "promotion_participation_count", "average_basket_size",
        "use_count", "reward_points_used" ]

    X = df[input_features]

    # Identificamos columnas numéricas y categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

    logger.info(f"Columnas numéricas: {numeric_features}")
    logger.info(f"Columnas categóricas: {categorical_features}")

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    hyperparams = hyperparams or {'n_estimators': 100, 'max_depth': 10}
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**hyperparams))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Split de datos completado: {X_train.shape[0]} ejemplos de entrenamiento")

    logger.info("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    logger.info(f"Accuracy entrenamiento: {train_accuracy:.4f}, Accuracy prueba: {test_accuracy:.4f}")

    joblib.dump(pipeline, output_path)
    logger.info(f"Modelo guardado en {output_path}")

    return {
        "metrics": {
            "accuracy_train": float(train_accuracy),
            "accuracy_test": float(test_accuracy)
        },
        "hyperparameters": hyperparams,
        "n_features": X.shape[1],
        "n_samples_train": X_train.shape[0],
        "feature_names": X.columns.tolist()
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    train_model(args.data_path, args.output_path)

