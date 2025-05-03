# Etapa de desarrollo
FROM python:3.10-slim AS development

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la app (incluyendo scripts de entrenamiento)
COPY . .

# Crear directorio para modelos (si es necesario)
RUN mkdir -p models

# Exponer puerto
EXPOSE 8000

# Variables de entorno
ENV PYTHONPATH=/app \
    DATA_PATH=/app/data/models/membership_groceries_userprofile.csv

# Etapa de producción (aquí podrías entrenar el modelo si es necesario)
FROM development AS production

# Comando para iniciar la API
CMD ["uvicorn", "app.models.main:app", "--host", "0.0.0.0", "--port", "8000"]