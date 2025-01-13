import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect
from PIL import Image
import mediapipe as mp
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import tensorflow as tf

# Inicializar Flask
app = Flask(__name__)

# Configuración de Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDS_FILE = 'credentials.json'
DATASET_FOLDER_ID = '14asAhgF9Y4GdWAlDbYSqsN0FlxYP_IJg'

# Inicializar Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Puntos clave principales
IMPORTANT_POINTS = {
    'left_eyebrow': [70, 107],
    'right_eyebrow': [336, 300],
    'nose': [1],
    'mouth': [78, 308, 13, 14],
    'left_eye': [33, 159, 133],
    'right_eye': [362, 386, 263],
}

# Diccionario para convertir etiquetas en texto
label_to_text = {
    0: 'Enojo',
    1: 'Disgusto',
    2: 'Miedo',
    3: 'Felicidad',
    4: 'Tristeza',
    5: 'Sorpresa',
    6: 'Neutral'
}

# Cargar el modelo de emociones entrenado
def load_emotion_model():
    return tf.keras.models.load_model('emotion_model_icml.h5')

emotion_model = load_emotion_model()

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
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
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
    try:
        img = Image.open(file).convert('L')  # Escala de grises
        img.thumbnail((800, 800))
        img_arr = np.array(img)

        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(np.stack((img_arr,) * 3, axis=-1))  # Convertir a RGB

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        def preprocess_face(image_array, landmarks):
            h, w = image_array.shape[:2]
            x_min = int(min([lm.x for lm in landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in landmarks.landmark]) * h)

            # Recortar y ajustar el rostro
            face = image_array[max(0, y_min):min(h, y_max), max(0, x_min):min(w, x_max)]
            if face.size == 0:
                raise ValueError("No se pudo recortar correctamente el rostro.")
            face_resized = Image.fromarray(face).resize((48, 48)).convert('L')
            face_gray = np.array(face_resized) / 255.0
            return np.expand_dims(np.expand_dims(face_gray, axis=-1), axis=0)

        def draw_landmarks(image_array, landmarks, emotion_label=None):
            h, w = image_array.shape
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(image_array, cmap='gray')
            ax.axis('off')

            for part, indices in IMPORTANT_POINTS.items():
                for idx in indices:
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    ax.plot(x, y, 'rx', markersize=4)

            if emotion_label:
                ax.set_title(emotion_label, fontsize=16, color='red')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close()
            return buf

        face_crop = preprocess_face(img_arr, face_landmarks)
        emotion_prediction = emotion_model.predict(face_crop)
        max_confidence = np.max(emotion_prediction)
        emotion_label_index = np.argmax(emotion_prediction)

        if max_confidence < 0.5:
            emotion_label = f"Insegura: {os.path.splitext(file.filename)[0].capitalize()}"
        else:
            emotion_label = label_to_text.get(emotion_label_index, "Desconocida")

        original_buf = draw_landmarks(img_arr, face_landmarks, emotion_label=emotion_label)
        flipped_buf = draw_landmarks(np.flip(img_arr, axis=1), face_landmarks, emotion_label=emotion_label)
        brightened_buf = draw_landmarks(np.clip(img_arr * 1.5, 0, 255).astype(np.uint8), face_landmarks, emotion_label=emotion_label)
        rotated_buf = draw_landmarks(np.flipud(img_arr), face_landmarks, emotion_label=emotion_label)  # Rotar hacia abajo

        def buffer_to_base64(buffer, filename):
            content = buffer.getvalue()
            upload_to_drive(content, filename)
            img_base64 = base64.b64encode(content).decode('utf-8')
            buffer.close()
            return f"data:image/png;base64,{img_base64}"

        return {
            'original': buffer_to_base64(original_buf, "original.png"),
            'flipped': buffer_to_base64(flipped_buf, "flipped.png"),
            'brightened': buffer_to_base64(brightened_buf, "brightened.png"),
            'rotated': buffer_to_base64(rotated_buf, "rotated.png"),
        }
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)



