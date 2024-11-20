import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect
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


def upload_to_drive(file_content, filename):
    """Sube un archivo a Google Drive."""
    media = MediaIoBaseUpload(io.BytesIO(file_content), mimetype='image/png')
    drive_service.files().create(
        body={'name': filename, 'parents': [DATASET_FOLDER_ID]},
        media_body=media,
        fields='id'
    ).execute()


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
        images = process_and_plot_image(file)
        if images:
            return render_template(
                'index.html',
                original_image=images['original'],
                flipped_image=images['flipped'],
                brightened_image=images['brightened'],
                rotated_image=images['rotated']
            )
        else:
            return render_template('index.html', error="No se detectó ningún rostro en la imagen.")


def process_and_plot_image(file):
    """Procesa la imagen y genera cuatro versiones: original, girada, con brillo y rotada."""
    img = Image.open(file).convert('L')
    img.thumbnail((800, 800))
    img_arr = np.array(img)

    faces = face_detector(img_arr)
    if len(faces) == 0:
        return None

    face = faces[0]
    landmarks = landmark_predictor(img_arr, face)

    # Dibujar puntos faciales en la imagen
    def draw_landmarks(image_array, landmarks, transform=None):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image_array, cmap='gray')
        ax.axis('off')

        key_points = [17, 21, 22, 26, 36, 39, 37, 42, 45, 43, 30, 48, 54, 51, 57]
        for n in key_points:
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Aplicar transformaciones para imágenes rotadas/volteadas
            if transform == "flipped":
                x = image_array.shape[1] - x
            elif transform == "rotated":
                x = image_array.shape[1] - x
                y = image_array.shape[0] - y

            ax.plot(x, y, 'rx', markersize=5)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        return buf

    def buffer_to_base64(buffer, filename):
        content = buffer.getvalue()
        upload_to_drive(content, filename)
        img_base64 = base64.b64encode(content).decode('utf-8')
        buffer.close()
        return f"data:image/png;base64,{img_base64}"

    # Generar imágenes procesadas
    original_buf = draw_landmarks(img_arr, landmarks)
    flipped_buf = draw_landmarks(img_arr[:, ::-1], landmarks, transform="flipped")
    brightened_buf = draw_landmarks(np.clip(img_arr * 1.8, 0, 255).astype(np.uint8), landmarks)
    rotated_buf = draw_landmarks(np.rot90(img_arr, 2), landmarks, transform="rotated")

    return {
        'original': buffer_to_base64(original_buf, "original.png"),
        'flipped': buffer_to_base64(flipped_buf, "flipped.png"),
        'brightened': buffer_to_base64(brightened_buf, "brightened.png"),
        'rotated': buffer_to_base64(rotated_buf, "rotated.png"),
    }


if __name__ == '__main__':
    app.run(debug=True, port=5003)
