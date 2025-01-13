import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar y preprocesar datos desde icml_face_data.csv
def load_icml_data(file_path):
    """Carga datos desde icml_face_data.csv."""
    data = pd.read_csv(file_path)
    pixels = data[' pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))  # Convertir a array
    X = np.stack(pixels.values) / 255.0  # Normalizar los píxeles
    X = X.reshape(-1, 48, 48, 1)  # Reorganizar la forma de las imágenes
    y = tf.keras.utils.to_categorical(data['emotion'], num_classes=7)  # Ajustar según el número de clases
    return X, y

# Ruta al archivo icml_face_data.csv
csv_file_path = "icml_face_data.csv"
X, y = load_icml_data(csv_file_path)

# Dividir datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Ajustar brillo
    shear_range=0.15,  # Shear para mayor distorsión
    fill_mode='nearest'  # Rellenar los píxeles al hacer transformaciones
)
datagen.fit(X_train)

# Construcción del modelo
def create_emotion_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(7, activation='softmax')  # 7 clases de emociones
    ])
    return model

# Crear y compilar el modelo
model = create_emotion_model()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Ajustar la tasa de aprendizaje si no mejora el rendimiento
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=1e-6)

# Entrenamiento del modelo
batch_size = 64
epochs = 50
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=[lr_reduction]
)

# Evaluar el modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Precisión en el conjunto de prueba: {test_accuracy:.2f}")

# Guardar el modelo
model.save('emotion_model_icml.h5')
print("Modelo guardado como 'emotion_model_icml.h5'")

# Evaluación del rendimiento
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Reporte de clasificación
label_to_text = {
    0: 'Enojo', 1: 'Disgusto', 2: 'Miedo', 3: 'Felicidad', 
    4: 'Tristeza', 5: 'Sorpresa', 6: 'Neutral'
}
print("\nReporte de clasificación:\n", classification_report(y_true_classes, y_pred_classes, target_names=list(label_to_text.values())))

# Matriz de confusión
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_to_text.values()), yticklabels=list(label_to_text.values()))
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')
plt.show()

# Gráficas de entrenamiento
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión durante el entrenamiento')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento')
plt.show()
