import pandas as pd
import cv2
import numpy as np
import urllib.request
import face_recognition
import os
from datetime import datetime

from PIL import ImageGrab

path = 'image_folder/'
image_paths = [os.path.join(path, image_name) for image_name in os.listdir(path)]

url='http://192.168.0.145/cam.jpg'

if 'Attendance.csv' in os.listdir(os.path.join(os.getcwd())):
    print("there iss..")
    os.remove("Attendance.csv")
else:
    df=pd.DataFrame(list())
    df.to_csv("Attendance.csv")
    
 
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

print(images)  # Asegúrate de que esta lista no esté vacía


def findEncodings(images):
    encodeList = []

    for imgPath in images:  # Asegúrate de que `images` contenga las rutas de las imágenes
        img = cv2.imread(imgPath)  # Cargar la imagen desde la ruta

        if img is None:  # Verifica si la imagen fue cargada correctamente
            print(f"No se pudo cargar la imagen: {imgPath}")
            continue  # Salta al siguiente archivo si la imagen no se puede cargar
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Si se cargó correctamente, convierte el color
        encode = face_recognition.face_encodings(img)[0]  # Procesa la imagen para encontrar las caras
        encodeList.append(encode)  # Agrega la codificación de la cara a la lista

    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()


        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM

def captureScreen(bbox=(300,300,690+300,530+300)):
     capScr = np.array(ImageGrab.grab(bbox))
     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
     return capScr

encodeListKnown = findEncodings(image_paths)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgnp,-1)

    img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1)

    if key==ord('q'):
        break
    
    cv2.destroyAllWindows()
    cv2.imread
