import os
import cv2
import pickle
import numpy as np
import urllib.request
import face_recognition
from datetime import datetime
import pandas as pd

ENCODINGS_FILE = 'faces_encodings.txt'  # Archivo txt para guardar codificaciones
ATTENDANCE_FILE = 'Attendance.csv'  # Archivo CSV para guardar asistencias

if not os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump([], f)

if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=['Name', 'Time'])
    df.to_csv(ATTENDANCE_FILE, index=False)

# URL de la cámara ESP32
ESP32_CAM_URL = 'http://192.168.0.145/cam-hi.jpg'

# Función para cargar codificaciones desde el archivo

def load_encodings():
    with open(ENCODINGS_FILE, 'rb') as f:
        return pickle.load(f)

# Función para guardar codificaciones en el archivo

def save_encodings(encodings):
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(encodings, f)

# Cargar codificaciones conocidas
known_encodings = load_encodings()
class_names = [enc["name"] for enc in known_encodings]

# Función para añadir un nuevo visitante
def add_visitor(image_url, name):
    global known_encodings
    try:
        # Descargar la imagen
        resp = urllib.request.urlopen(image_url)
        img_array = np.array(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        # Codificar la imagen
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)

        if encodings:
            known_encodings.append({"name": name, "encoding": encodings[0]})
            save_encodings(known_encodings)
            print(f"Visitante {name} añadido correctamente.")
        else:
            print("No se detectaron rostros en la imagen.")

    except Exception as e:
        print(f"Error al añadir visitante: {e}")

# Función para eliminar un visitante
def remove_visitor(name):
    global known_encodings
    updated_encodings = [enc for enc in known_encodings if enc["name"] != name]

    if len(updated_encodings) < len(known_encodings):
        known_encodings = updated_encodings
        save_encodings(known_encodings)
        print(f"Visitante {name} eliminado correctamente.")
    else:
        print(f"Visitante {name} no encontrado.")

# Función para registrar asistencia
def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    if name not in df['Name'].values:
        now = datetime.now().strftime('%H:%M:%S')
        df = df.append({"Name": name, "Time": now}, ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"Asistencia registrada para {name}.")
    else:
        print(f"{name} ya tiene asistencia registrada.")

# Procesamiento principal
while True:
    try:
        # Obtener imagen de la cámara ESP32-CAM
        img_resp = urllib.request.urlopen(ESP32_CAM_URL)
        img_array = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)

        # Redimensionar y convertir a RGB
        frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        # Detectar rostros en el frame actual
        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                [enc["encoding"] for enc in known_encodings], face_encoding
            )
            face_distances = face_recognition.face_distance(
                [enc["encoding"] for enc in known_encodings], face_encoding
            )

            if matches:
                best_match_index = np.argmin(face_distances)
                name = known_encodings[best_match_index]["name"]

                y1, x2, y2, x1 = [v * 4 for v in face_location]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)

        cv2.imshow('ESP32-CAM', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error en el procesamiento: {e}")
        break


cv2.destroyAllWindows()