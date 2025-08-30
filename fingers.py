import cv2

import mediapipe as mp



mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils



# Función para determinar si un dedo está abierto o cerrado

def dedo_abierto(hand_landmarks, dedo, imagen_ancho, imagen_alto):

  """

  dedo: índice del dedo a verificar

  Devuelve True si el dedo está abierto, False si está cerrado.

  """

  # Landmark indices for fingertips and pip joints:

  tips = [4, 8, 12, 16, 20]

  pip_joints = [3, 6, 10, 14, 18]



  # El pulgar es especial: para el pulgar, comparamos x, para otros dedos, comparamos y

  if dedo == 0:

    # Pulgar: abierto si punta del pulgar está más a la derecha que la base (mano derecha)

    return hand_landmarks.landmark[tips[dedo]].x > hand_landmarks.landmark[pip_joints[dedo]].x

  else:

    # Otros dedos: abierto si punta está arriba (menor y) que la articulación pip

    return hand_landmarks.landmark[tips[dedo]].y < hand_landmarks.landmark[pip_joints[dedo]].y



cap = cv2.VideoCapture(0)



with mp_hands.Hands(

  max_num_hands=1, # Solo una mano para simplificar

  min_detection_confidence=0.7,

  min_tracking_confidence=0.7) as hands:



  while True:

    ret, frame = cap.read()

    if not ret:

      break



    frame = cv2.flip(frame, 1) # Espejo

    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)



    numero = None



    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:

        mp_drawing.draw_landmarks(

          frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)



        dedos_abiertos = [dedo_abierto(hand_landmarks, i, w, h) for i in range(5)]

        # Contar dedos abiertos

        cantidad_dedos = sum(dedos_abiertos)



        numero = cantidad_dedos



        # Mostrar número detectado en pantalla

        cv2.putText(frame, f'Numero detectado: {numero}', (10, 50),

              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)



    else:

      cv2.putText(frame, 'Mano no detectada', (10, 50),

            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)



    cv2.imshow("Deteccion de numeros con mano", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):

      break



cap.release()

cv2.destroyAllWindows()
