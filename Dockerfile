FROM python:3.8-slim

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios
COPY . /app

# Actualizar e instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    cmake \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar pip y las dependencias de Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Exponer el puerto que usa Flask
EXPOSE 5000

# Comando para iniciar la aplicación
CMD ["python", "app.py"]



