FROM python:3.12-slim




RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-dejavu-core \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Crea carpeta para la app
WORKDIR /app

# Copia tus archivos
COPY . /app

# Instala las dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto
EXPOSE 8080

# Comando para iniciar tu app Flask
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8080}"]

