import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect
from PIL import Image
import dlib  # Dlib para puntos faciales
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload  # Asegúrate de importar esto

# Inicializar la aplicación Flask
app = Flask(__name__)

# Configuración de Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDS_FILE = 'credentials.json'
DATASET_FOLDER_ID = '14asAhgF9Y4GdWAlDbYSqsN0FlxYP_IJg'  # ID de la carpeta de Google Drive donde guardarás los archivos

# Ruta al predictor de puntos faciales
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Inicializar Dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)

def authenticate_drive():
    """Autentica y devuelve el servicio de Google Drive."""
    # Eliminar cualquier token guardado previamente para forzar la autenticación cada vez
    if os.path.exists('token.json'):
        os.remove('token.json')

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
        image_base64, landmarks = process_image(file)
        if landmarks:
            save_landmarks_to_csv(landmarks)
            upload_to_drive('landmarks.csv', 'text/csv')
        return render_template('index.html', image_file=image_base64)

def process_image(file):
    """Procesa la imagen y dibuja puntos faciales."""
    img = Image.open(file).convert('L')
    img.thumbnail((800, 800))
    img_arr = np.array(img)

    faces = face_detector(img_arr)
    if len(faces) == 0:
        print("No se detectó ningún rostro.")
        return None, None

    face = faces[0]
    landmarks = landmark_predictor(img_arr, face)

    plt.figure(figsize=(img.width / 100, img.height / 100))
    plt.imshow(img_arr, cmap='gray')
    plt.axis('off')

    key_points = [17, 21, 22, 26, 36, 39, 37, 42, 45, 43, 30, 48, 54, 51, 57]
    x_coords, y_coords = [], []

    for n in key_points:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        plt.plot(x, y, 'rx', markersize=5)
        x_coords.append(x)
        y_coords.append(y)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    save_image_to_drive(buf, 'processed_image.png')
    return f"data:image/png;base64,{img_base64}", list(zip(x_coords, y_coords))

def save_landmarks_to_csv(landmarks):
    """Guarda los puntos faciales en un CSV."""
    if os.path.exists('landmarks.csv'):
        df = pd.read_csv('landmarks.csv')
    else:
        df = pd.DataFrame(columns=['X', 'Y'])

    new_data = pd.DataFrame(landmarks, columns=['X', 'Y'])
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv('landmarks.csv', index=False)

def upload_to_drive(filename, mimetype):
    """Sube un archivo a Google Drive."""
    file_metadata = {'name': filename, 'parents': [DATASET_FOLDER_ID]}
    media = MediaFileUpload(filename, mimetype=mimetype)
    drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

def save_image_to_drive(image_buffer, filename):
    """Guarda una imagen en Google Drive."""
    file_metadata = {'name': filename, 'parents': [DATASET_FOLDER_ID]}
    media = MediaIoBaseUpload(image_buffer, mimetype='image/png')
    drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

if __name__ == '__main__':
    app.run(debug=True, port=5002)
