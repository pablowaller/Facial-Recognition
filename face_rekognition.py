import firebase_admin
from firebase_admin import credentials, storage, db
import urllib.request
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import requests
import urllib.parse
import time  

cred = credentials.Certificate("firebase_credentials.json")  
firebase_admin.initialize_app(cred, {
    'storageBucket': 'sense-bell.firebasestorage.app',  
    'databaseURL': 'https://sense-bell-default-rtdb.firebaseio.com/' 
})

bucket = storage.bucket()

ESP_IP = "192.168.0.145"  
IMAGE_URL = f"http://{ESP_IP}/cam-hi.jpg"
DETECTION_INTERVAL = 300  

last_detection_time = {}

def download_images_from_firebase():
    blobs = bucket.list_blobs(prefix="photos/")  
    images = []
    classNames = []

    for blob in blobs:
        if blob.name.endswith(('.jpg', '.png', '.jpeg')):
            encoded_name = urllib.parse.quote(blob.name, safe='')
            url = f"https://firebasestorage.googleapis.com/v0/b/sense-bell.firebasestorage.app/o/{encoded_name}?alt=media"
            try:
                resp = urllib.request.urlopen(url)
                img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                images.append(img)
                classNames.append(blob.name.split("/")[-1].split(".")[0]) 
            except Exception as e:
                print(f"❌ Error al descargar {blob.name}: {e}")

    return images, classNames

print("🔹 Descargando imágenes desde Firebase...")
images, classNames = download_images_from_firebase()
print(f"🔹 Se cargaron {len(images)} imágenes para reconocimiento.")

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  
            encodeList.append(encode[0])  
    return encodeList

encodeListKnown = findEncodings(images)
print("🔹 Codificación facial completada.")

attendance_file = 'Attendance.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Time\n')
    print("🔹 Archivo de asistencia creado.")

def markAttendance(name):
    now = datetime.now()
    if name in last_detection_time:
        time_elapsed = now - last_detection_time[name]
        if time_elapsed.total_seconds() < DETECTION_INTERVAL:
            return

    timeString = now.strftime('%H:%M:%S')
    with open(attendance_file, 'a') as f:
        f.write(f'{name},{timeString}\n')
    print(f"✅ Asistencia registrada para {name} a las {timeString}")
    markAttendanceInFirebase(name, timeString)

    last_detection_time[name] = now

def get_priority_from_firebase(name):
    visitors_url = "https://sense-bell-default-rtdb.firebaseio.com/visitors.json"
    try:
        response = requests.get(visitors_url)
        if response.status_code == 200:
            visitors_data = response.json()
            for key, visitor in visitors_data.items():
                if visitor.get("name") == name:
                    return visitor.get("priority", 0)  
        else:
            print(f"❌ Error al obtener visitantes: {response.text}")
    except Exception as e:
        print(f"❌ Error en get_priority_from_firebase: {e}")
    return 0

def markAttendanceInFirebase(name, timestamp):
    priority = get_priority_from_firebase(name)
    url = "https://sense-bell-default-rtdb.firebaseio.com/attendance.json"
    data = {
        "name": name,
        "timestamp": timestamp,
        "priority": priority, 
        "message": f"{name} está en la puerta!"
    }
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print(f"✅ Asistencia subida a Firebase para {name} con prioridad {priority}")
    else:
        print(f"❌ Error subiendo asistencia a Firebase: {response.text}")

def analyze_high_res_image():
    try:
        print("🔹 Descargando imagen de ESP32-CAM...")
        resp = urllib.request.urlopen(IMAGE_URL)
        img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("❌ Error: No se pudo decodificar la imagen")
            return

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(img_rgb)
        encodesCurFrame = face_recognition.face_encodings(img_rgb, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis) if faceDis.size > 0 else None

            if matchIndex is not None and matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(f"👤 Persona detectada: {name}")  
                markAttendance(name)

                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("ESP32-CAM High-Res", frame)
        cv2.waitKey(5000)  
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ Error al analizar la imagen: {e}")

while True:
    analyze_high_res_image()
    time.sleep(5)  
