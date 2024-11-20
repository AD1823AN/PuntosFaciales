# Usa una imagen base de Python
FROM python:3.9-slim

# Instalar dependencias del sistema necesarias para dlib y matplotlib
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    wget \
    unzip \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libfreetype6-dev \
    pkg-config \
    libpng-dev \
    libjpeg-dev \
    && apt-get clean


# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . .

# Actualizar pip y wheel
RUN pip install --upgrade pip wheel setuptools

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import matplotlib.pyplot as plt" && echo 'Matplotlib installed correctly' || exit 1



# Exponer el puerto 5000 (usado por Flask)
EXPOSE 5000

# Comando para iniciar la aplicación
CMD ["python", "app.py"]

