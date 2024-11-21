# Usa una imagen base de Python
FROM python:3.9-slim

# Instalar dependencias del sistema necesarias para Mediapipe y matplotlib
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgtk-3-dev \
    && apt-get clean

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . .

# Actualizar pip y wheel
RUN pip install --upgrade pip wheel setuptools

# Instalar dependencias de Python, incluyendo Mediapipe
RUN pip install --no-cache-dir -r requirements.txt

# Verificar la instalación de Mediapipe y Matplotlib
RUN python -c "import mediapipe as mp; import matplotlib.pyplot as plt" && echo 'Mediapipe y Matplotlib instalados correctamente' || exit 1

# Exponer el puerto 5000 (usado por Flask)
EXPOSE 5000

# Comando para iniciar la aplicación
CMD ["python", "app.py"]


