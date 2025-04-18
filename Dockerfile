# Etapa de desarrollo
FROM python:3.10-slim AS development
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Etapa de producción
FROM python:3.10-slim AS production
WORKDIR /app
COPY --from=development /app/app ./app
COPY --from=development /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models ./models

# Establecer variables de entorno
ENV MODEL_PATH=/app/models/grocery_membership_model.joblib
ENV PYTHONPATH=/app

# Exponer puerto para API
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]