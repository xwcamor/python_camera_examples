import cv2



# Clasificador HaarCascade para ojos

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")



# Iniciar cámara

cap = cv2.VideoCapture(0)



if not cap.isOpened():

  print("❌ No se pudo abrir la cámara")

  exit()



while True:

  ret, frame = cap.read()

  if not ret:

    print("❌ No se pudo leer el frame")

    break



  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



  # Detectar ojos en la imagen completa

  eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)



  for (ex, ey, ew, eh) in eyes:

    cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # Rectángulo verde



  cv2.imshow("Detección de Ojos", frame)



  if cv2.waitKey(1) & 0xFF == ord('q'):

    break



cap.release()

cv2.destroyAllWindows()

