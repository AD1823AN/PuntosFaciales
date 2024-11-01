import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import dlib
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Inicializar Flask
app = Flask(__name__)

# Configuración de Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDS_FILE = 'credentials.json'
DATASET_FOLDER_ID = '14asAhgF9Y4GdWAlDbYSqsN0FlxYP_IJg'  # ID de la carpeta de Drive

# Ruta al predictor de puntos faciales
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Inicializar Dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)

def authenticate_drive():
    """Autentica y devuelve el servicio de Google Drive."""
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

drive_service = authenticate_drive()

@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Maneja la subida de archivos y procesa la imagen."""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        image_base64 = process_and_plot_image(file)
        if image_base64:
            return render_template('index.html', image_file=image_base64)
        else:
            # Mensaje de error si no se detectó rostro
            return render_template('index.html', error="No se detectó ningún rostro en la imagen.")

def process_and_plot_image(file):
    """Procesa la imagen y dibuja los puntos faciales."""
    img = Image.open(file).convert('L')
    img.thumbnail((800, 800))
    img_arr = np.array(img)

    faces = face_detector(img_arr)
    if len(faces) == 0:
        print("No se detectó ningún rostro.")
        return None

    face = faces[0]
    landmarks = landmark_predictor(img_arr, face)

    # Configurar la figura para mostrar puntos faciales
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_arr, cmap='gray')
    ax.axis('off')  # Oculta los ejes para una presentación limpia

    # Dibuja puntos faciales relevantes
    key_points = [17, 21, 22, 26, 36, 39, 37, 42, 45, 43, 30, 48, 54, 51, 57]
    for n in key_points:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        ax.plot(x, y, 'rx', markersize=5)

    # Guardar la gráfica en un buffer y convertir a base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    # Subir la imagen procesada a Google Drive
    save_image_to_drive(buf, 'processed_image.png')

    # Codificar la imagen en base64 para mostrar en la web
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()  # Cierra el buffer
    return f"data:image/png;base64,{img_base64}"

def save_image_to_drive(image_buffer, filename):
    """Guarda una imagen en Google Drive."""
    file_metadata = {'name': filename, 'parents': [DATASET_FOLDER_ID]}
    media = MediaIoBaseUpload(image_buffer, mimetype='image/png')
    drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

if __name__ == '__main__':
    app.run(debug=True, port=5002)
