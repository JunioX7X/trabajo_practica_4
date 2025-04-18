
FROM python:3.10-slim AS builder

WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Etapa de producción
FROM python:3.10-slim AS production

WORKDIR /app

# Copiar solo los archivos necesarios desde la etapa de compilación
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar código de aplicación
COPY app ./app
COPY models ./models

# Configurar variables de entorno
ARG API_KEY
ENV API_KEY=${API_KEY}
ENV MODEL_PATH=/data/membership_groceries_userprofile.csv
ENV PYTHONPATH=/data

# Exponer puerto para API
EXPOSE 8000

# Crear script de punto de entrada
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'if [ "$SERVICE_TYPE" = "api" ]; then' >> /app/entrypoint.sh && \
    echo '  uvicorn app.api.main:app --host 0.0.0.0 --port 8000' >> /app/entrypoint.sh && \
    echo 'elif [ "$SERVICE_TYPE" = "streamlit" ]; then' >> /app/entrypoint.sh && \
    echo '  streamlit run app/api/streamlit_app.py --server.port 8501 --server.address 0.0.0.0' >> /app/entrypoint.sh && \
    echo 'else' >> /app/entrypoint.sh && \
    echo '  echo "Por favor establece SERVICE_TYPE a api o streamlit"' >> /app/entrypoint.sh && \
    echo '  exit 1' >> /app/entrypoint.sh && \
    echo 'fi' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Ejecutar script de punto de entrada
ENV SERVICE_TYPE=api
ENTRYPOINT ["/app/entrypoint.sh"]
