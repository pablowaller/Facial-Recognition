import threading
import firebase_admin
from firebase_admin import credentials, storage, db
import urllib.request
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import requests
import urllib.parse
import time

cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'sense-bell.firebasestorage.app',
    'databaseURL': 'https://sense-bell-default-rtdb.firebaseio.com/'
})

priority_low_ref = db.reference('doorbell/priority_low')
priority_medium_ref = db.reference('doorbell/priority_medium')
priority_high_ref = db.reference('doorbell/priority_high')
ip_ref = db.reference('currentIP')
new_face_ref = db.reference('new_face_flag')  

last_update_time = time.time()
UPDATE_INTERVAL = 30  

bucket = storage.bucket()
DETECTION_INTERVAL = 300
last_detection_time = {}

def get_esp32cam_url():
    """Obtiene la URL del stream de la ESP32-CAM desde Firebase"""
    try:
        ip_data = ip_ref.get()
        if ip_data and 'ip' in ip_data:
            ip = ip_data['ip']
            return f"http://{ip}/stream"
        return "http://192.168.0.145/stream"  # Fallback IP
    except Exception as e:
        print(f"❌ Error al obtener IP de Firebase: {e}")
        return "http://192.168.0.145/stream"  # Fallback IP 

def select_camera_source():
    """Permite al usuario seleccionar la fuente de video"""
    print("\nSeleccione la cámara a utilizar:")
    print("1. Webcam")
    print("2. ESP32-CAM")
    
    while True:
        choice = input("Ingrese su opción (1 o 2): ")
        if choice in ['1', '2']:
            return int(choice)
        print("Opción inválida. Intente nuevamente.")

def initialize_video_source(choice):
    """Inicializa la fuente de video según la selección del usuario"""
    if choice == 1:  
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Webcam iniciada.")
            return cap, 'Webcam Reconocimiento Facial'
        else:
            print("❌ No se pudo abrir la webcam.")
            return None, None
    else:  
        stream_url = get_esp32cam_url()
        print(f"🔗 Intentando conectar a: {stream_url}")
        
        # Try opening with OpenCV first
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            print("✅ Transmisión de ESP32-CAM iniciada con OpenCV.")
            return cap, 'ESP32-CAM Reconocimiento Facial'
        
        # If OpenCV fails, try with urllib
        try:
            resp = urllib.request.urlopen(stream_url)
            bytes = bytes()
            while True:
                bytes += resp.read(1024)
                a = bytes.find(b'\xff\xd8')
                b = bytes.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes[a:b+2]
                    bytes = bytes[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        print("✅ Transmisión de ESP32-CAM iniciada con urllib.")
                        # Create a dummy VideoCapture object
                        cap = cv2.VideoCapture()
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, img.shape[1])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img.shape[0])
                        return cap, 'ESP32-CAM Reconocimiento Facial'
        except Exception as e:
            print(f"❌ Error al conectar a ESP32-CAM: {e}")
            return None, None

def download_images_from_firebase():
    """Descarga imágenes de Firebase optimizada (solo si hay cambios)"""
    try:
        blobs = list(bucket.list_blobs(prefix="photos/"))
        if not blobs:
            return [], []
        
        latest_blob = max(blobs, key=lambda x: x.updated)
        if hasattr(download_images_from_firebase, 'last_updated') and \
           latest_blob.updated.timestamp() <= download_images_from_firebase.last_updated:
            return [], []
        
        download_images_from_firebase.last_updated = time.time()
        images = []
        classNames = []

        for blob in blobs:
            if blob.name.endswith(('.jpg', '.png', '.jpeg')):
                try:
                    encoded_name = urllib.parse.quote(blob.name, safe='')
                    url = f"https://firebasestorage.googleapis.com/v0/b/sense-bell.firebasestorage.app/o/{encoded_name}?alt=media"
                    resp = urllib.request.urlopen(url)
                    img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        images.append(img)
                        classNames.append(os.path.splitext(blob.name.split("/")[-1])[0])
                except Exception as e:
                    print(f"❌ Error al procesar {blob.name}: {e}")

        return images, classNames
    except Exception as e:
        print(f"❌ Error en descarga de imágenes: {e}")
        return [], []

images, classNames = download_images_from_firebase()

def findEncodings(images):
    """Genera codificaciones faciales para las imágenes"""
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

encodeListKnown = findEncodings(images)

attendance_file = 'Attendance.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Time\n')
    print("🔹 Archivo de asistencia creado.")

def markAttendance(name):
    """Registra la asistencia con control de intervalo"""
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
    activatePriorityForVisitor(name)

    last_detection_time[name] = now

def markAttendanceInFirebase(name, timestamp):
    """Envía la asistencia a Firebase"""
    url = "https://sense-bell-default-rtdb.firebaseio.com/attendance.json"
    data = {
        "name": name,
        "timestamp": timestamp,
        "message": f"{name} está en la puerta!"
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"✅ Asistencia subida a Firebase para {name}")
        else:
            print(f"❌ Error subiendo asistencia: {response.text}")
    except Exception as e:
        print(f"❌ Error de conexión con Firebase: {e}")

def activatePriorityForVisitor(name):
    """Activa la prioridad según el visitante"""
    try:
        priority = get_priority_from_firebase(name)
        print(f"⚡ Activando prioridad: {priority} para {name}")
        priority_low_ref.set(False)
        priority_medium_ref.set(False)
        priority_high_ref.set(False)
        
        if priority == 'low':
            priority_low_ref.set(True)
            print("🔴 Prioridad Baja activada")
        elif priority == 'medium':
            priority_medium_ref.set(True)
            print("🟡 Prioridad Media activada")
        elif priority == 'high':
            priority_high_ref.set(True)
            print("🟢 Prioridad Alta activada")
        else:
            print("ℹ️ Prioridad no definida, usando baja por defecto")
            priority_low_ref.set(True)
    except Exception as e:
        print(f"❌ Error al activar prioridad: {e}")
        priority_low_ref.set(True)

def get_priority_from_firebase(name):
    """Obtiene la prioridad desde Firebase"""
    print(f"🔍 Buscando prioridad para: {name}") 
    visitors_url = "https://sense-bell-default-rtdb.firebaseio.com/visitors.json"
    try:
        response = requests.get(visitors_url)
        if response.status_code == 200:
            visitors_data = response.json() or {}
            for visitor in visitors_data.values():
                if visitor.get("name") == name:
                    priority = visitor.get("priority", "medium").lower()  
                    print(f"DEBUG: Prioridad encontrada para {name}: {priority}")  
                    if priority in ["low", "medium", "high"]:
                        return priority
                    return "medium" 
        return "low"  
    except Exception as e:
        print(f"❌ Error en get_priority_from_firebase: {e}")
        return "low"

def process_frame(frame):
    """Procesa cada frame del video"""
    global encodeListKnown, classNames, last_update_time
    
    if (time.time() - last_update_time > UPDATE_INTERVAL) or (new_face_ref.get() is True):
        try:
            new_face_ref.set(False)
            new_images, new_classNames = download_images_from_firebase()
            if new_images:
                new_encodings = findEncodings(new_images)
                if new_encodings:
                    encodeListKnown = new_encodings
                    classNames = new_classNames
                    print(f"🔄 Rostros actualizados. Total: {len(classNames)}")
            last_update_time = time.time()
        except Exception as e:
            print(f"⚠️ Error al actualizar rostros: {e}")

    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis) if faceDis.size > 0 else None

        if matchIndex is not None and matches[matchIndex]:
            name = classNames[matchIndex].upper()
            decoded_name = urllib.parse.unquote(name)
            
            # Dibujar rectángulo y nombre siempre que se detecte la cara
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, decoded_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Solo registrar asistencia si ha pasado el intervalo
            now = datetime.now()
            if decoded_name in last_detection_time:
                time_elapsed = (now - last_detection_time[decoded_name]).total_seconds()
                if time_elapsed < DETECTION_INTERVAL:
                    continue  
            
            print(f"👤 Persona detectada: {decoded_name}")  
            markAttendance(decoded_name)

    return frame

def video_capture_thread():
    """Hilo principal de captura de video"""
    global cap, window_name
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("❌ No se pudo leer el marco. Reintentando...")
                time.sleep(2)
                cap.release()
                choice = 2 if "ESP32" in window_name else 1
                cap, window_name = initialize_video_source(choice)
                if cap is None:
                    break
                continue

            processed_frame = process_frame(frame)
            cv2.imshow(window_name, processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"❌ Error en captura de video: {e}")
            time.sleep(2)
            if 'cap' in globals():
                cap.release()
            choice = 2 if "ESP32" in window_name else 1
            cap, window_name = initialize_video_source(choice)
            if cap is None:
                break

# Inicialización principal
if __name__ == "__main__":
    camera_choice = select_camera_source()
    cap, window_name = initialize_video_source(camera_choice)
    if cap is None:
        exit()

    video_thread = threading.Thread(target=video_capture_thread, daemon=True)
    video_thread.start()

    try:
        while video_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Programa terminado por el usuario")
    finally:
        if 'cap' in globals():
            cap.release()
        cv2.destroyAllWindows()
        print("✅ Programa terminado correctamente")
