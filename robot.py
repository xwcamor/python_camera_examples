descargar primero en el cmd             pip install mediapipe opencv-python           (tarda un rato)



RobotCara.py   

import os

import cv2

import mediapipe as mp



# 🔇 Desactivar mensajes de TensorFlow

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# Inicializar MediaPipe

mp_face_mesh = mp.solutions.face_mesh

mp_drawing = mp.solutions.drawing_utils



# Configuración de estilos

drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1) # verde robot



# Iniciar cámara

cap = cv2.VideoCapture(0)



with mp_face_mesh.FaceMesh(

  max_num_faces=5,

  refine_landmarks=True,

  min_detection_confidence=0.5,

  min_tracking_confidence=0.5) as face_mesh:



  while True:

    ret, frame = cap.read()

    if not ret:

      break



    # Convertir a RGB

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)



    if results.multi_face_landmarks:

      for face_landmarks in results.multi_face_landmarks:

        # Dibujar la malla facial verde

        mp_drawing.draw_landmarks(

          image=frame,

          landmark_list=face_landmarks,

          connections=mp_face_mesh.FACEMESH_TESSELATION,

          landmark_drawing_spec=drawing_spec,

          connection_drawing_spec=drawing_spec

        )



        # Dibujar "ojos rojos"

        h, w, _ = frame.shape

        left_eye = [33, 133] # índices aproximados del ojo izquierdo

        right_eye = [362, 263] # índices aproximados del ojo derecho



        for idx in left_eye + right_eye:

          x = int(face_landmarks.landmark[idx].x * w)

          y = int(face_landmarks.landmark[idx].y * h)

          cv2.circle(frame, (x, y), 5, (0, 0, 255), -1) # rojo



    # Mostrar en ventana

    cv2.imshow("Robot Cyborg Face", frame)



    # Salir con 'q'

    if cv2.waitKey(1) & 0xFF == ord('q'):

      break



cap.release()

cv2.destroyAllWindows()
