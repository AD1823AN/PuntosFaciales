# Imagen base de Python
FROM python:3.8-slim
# Establecer el directorio de trabajo en el contenedor
WORKDIR /app
# Copiar los archivos necesarios al contenedor
COPY . /app
# Actualizar el sistema y asegurar dependencias del sistema operativo
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean
# Instalar pip y las dependencias de Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt
# Exponer el puerto que usa Flask
EXPOSE 5000
# Comando para iniciar la aplicación
CMD ["python", "app.py"]


