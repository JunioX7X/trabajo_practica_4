# Etapa de desarrollo
FROM python:3.10-slim AS development

WORKDIR /app

# Copiar solo los archivos de requisitos primero para aprovechar caché de capas
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de aplicación
COPY .. ./app
COPY ../../data ./data

# Asegurar existencia del directorio de modelos
RUN mkdir -p models

# Exponer puerto para API
EXPOSE 8000

# Establecer variable de entorno para Python path
ENV PYTHONPATH=/app

# Etapa de producción
FROM development AS production



# Establecer variables específicas de producción
ENV MODEL_PATH=/app/models/grocery_membership_model.joblib \
    DATA_PATH=/app/data/membership_groceries_userprofile.csv

# Punto de entrada para servicio de API
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]