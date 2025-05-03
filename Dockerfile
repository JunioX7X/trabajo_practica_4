# Etapa de desarrollo
FROM python:3.10-slim AS development

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la app y datos
COPY app ./app
COPY data ./data

# Crear directorio para modelos
RUN mkdir -p models

# ENTRENAR EL MODELO durante el build
RUN python app/models/train.py \
    --data-path data/membership_groceries_userprofile.csv \
    --output-path /app/models/grocery_membership_model.joblib

RUN ls -lh /app/models/


# Exponer puerto
EXPOSE 8000

# Variables de entorno
ENV PYTHONPATH=/app

# Etapa de producción
FROM development AS production

CMD ["uvicorn", "app.models.main:app", "--host", "0.0.0.0", "--port", "8000"]

