import cv2



# Cargar clasificadores preentrenados

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')



# Captura de la cámara

cap = cv2.VideoCapture(0)



while True:

  ret, frame = cap.read()

  if not ret:

    break



  # Escala de grises para mejor detección

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



  # Detectar caras

  faces = face_cascade.detectMultiScale(gray, 1.3, 5)



  for (x, y, w, h) in faces:

    # Dibujar rectángulo alrededor de la cara

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)



    # Región de interés para buscar sonrisa dentro de la cara

    roi_gray = gray[y:y + h, x:x + w]

    roi_color = frame[y:y + h, x:x + w]



    # Detectar sonrisas

    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)



    if len(smiles) > 0:

      # Mostrar mensaje

      cv2.putText(frame, "¡Sonrisa detectada! Apagando camara...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

      cv2.imshow('Sonrisa', frame)

      cv2.waitKey(2000) # Espera 2 segundos para mostrar el mensaje

      cap.release()

      cv2.destroyAllWindows()

      print("✅ Cámara apagada por sonrisa.")

      exit()



  # Mostrar ventana de video

  cv2.imshow('Detección de Sonrisa', frame)



  # Salir con 'q'

  if cv2.waitKey(1) & 0xFF == ord('q'):

    break



# Liberar recursos

cap.release()

cv2.destroyAllWindows()
