import cv2

import mediapipe as mp

import numpy as np



# Cargar imágenes de ojos (sin canal alfa)

ojo_izquierdo_img = cv2.imread("ojo_izquierdo.png")

ojo_derecho_img = cv2.imread("ojo_derecho.png")



mp_face_mesh = mp.solutions.face_mesh



OJOS_DERECHO = [33, 133]

OJOS_IZQUIERDO = [362, 263]



def overlay_image(background, overlay, x, y, size):

  overlay_resized = cv2.resize(overlay, size)

  h, w, _ = overlay_resized.shape

  if y+h > background.shape[0] or x+w > background.shape[1] or x < 0 or y < 0:

    return background # Evitar error si sale del frame

  background[y:y+h, x:x+w] = overlay_resized

  return background



cap = cv2.VideoCapture(0)



with mp_face_mesh.FaceMesh(

  max_num_faces=1,

  refine_landmarks=True,

  min_detection_confidence=0.7,

  min_tracking_confidence=0.7

) as face_mesh:



  while True:

    ret, frame = cap.read()

    if not ret:

      break



    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



    results = face_mesh.process(rgb)



    if results.multi_face_landmarks:

      for face_landmarks in results.multi_face_landmarks:

        for idx, ojo_img in zip([OJOS_IZQUIERDO, OJOS_DERECHO],

                    [ojo_izquierdo_img, ojo_derecho_img]):

          x = int(face_landmarks.landmark[idx[0]].x * w)

          y = int(face_landmarks.landmark[idx[0]].y * h)

          x2 = int(face_landmarks.landmark[idx[1]].x * w)

          y2 = int(face_landmarks.landmark[idx[1]].y * h)



          ojo_ancho = max(30, abs(x2 - x))

          ojo_alto = int(ojo_ancho * ojo_img.shape[0] / ojo_img.shape[1])



          ojo_x = int((x + x2) / 2 - ojo_ancho / 2)

          ojo_y = int((y + y2) / 2 - ojo_alto / 2)



          if 0 <= ojo_x <= w - ojo_ancho and 0 <= ojo_y <= h - ojo_alto:

            frame = overlay_image(frame, ojo_img, ojo_x, ojo_y, (ojo_ancho, ojo_alto))



    cv2.imshow("Ojos reemplazados", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):

      break



cap.release()

cv2.destroyAllWindows()

