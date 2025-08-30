import cv2

import numpy as np

import mediapipe as mp



# ---------- Config ----------

MASK_PATH = "C:/Users/alu_torre1/anonymous-mask-115309810289b8xfjop0w.png" # cambia si usas otra ruta

CAM_INDEX = 0 # 0 = webcam principal



# Puntos de la malla de MediaPipe que usaremos

# Ojo derecho (externo), ojo izquierdo (externo), mentón

IDX_RIGHT_EYE = 33

IDX_LEFT_EYE = 263

IDX_CHIN   = 152



# Contorno "face oval" para recortar solo el rostro

FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,

         379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,

         234, 127, 162, 21, 54, 103, 67, 109]



# ---------- Carga de máscara ----------

mask_rgba = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)

if mask_rgba is None:

  raise FileNotFoundError(f"No pude cargar la máscara en: {MASK_PATH}")



# Asegurar canal alfa (si la PNG no lo tuviera)

if mask_rgba.shape[2] == 3:

  alpha = np.full(mask_rgba.shape[:2], 255, dtype=np.uint8)

  mask_rgba = np.dstack([mask_rgba, alpha])



h_m, w_m = mask_rgba.shape[:2]



# Puntos de control en la máscara (aprox. en proporciones del PNG)

# Ajustados para un encaje realista (ojos un poco por encima del centro y mentón abajo)

mask_pts = np.float32([

  (0.30 * w_m, 0.40 * h_m), # ojo derecho (desde la vista de la máscara)

  (0.70 * w_m, 0.40 * h_m), # ojo izquierdo

  (0.50 * w_m, 0.93 * h_m) # mentón

])



# ---------- MediaPipe ----------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(

  static_image_mode=False,

  refine_landmarks=True,

  max_num_faces=1,

  min_detection_confidence=0.6,

  min_tracking_confidence=0.6

)



cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():

  raise RuntimeError("No pude abrir la cámara. Cambia CAM_INDEX si tienes varias cámaras.")



def landmarks_to_points(landmarks, w, h, idxs):

  pts = []

  for i in idxs:

    lm = landmarks[i]

    pts.append((int(lm.x * w), int(lm.y * h)))

  return np.array(pts, dtype=np.int32)



def keep_only_face(frame, face_pts):

  # Crea una máscara negra y rellena el óvalo facial en blanco

  face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

  cv2.fillConvexPoly(face_mask, face_pts, 255)

  # Aplica la máscara al frame para dejar solo la cara

  face_only = cv2.bitwise_and(frame, frame, mask=face_mask)

  # Fondo negro

  bg_black = np.zeros_like(frame)

  bg_black = cv2.bitwise_and(bg_black, bg_black, mask=255 - face_mask)

  return face_only + bg_black, face_mask



def overlay_affine(base_bgr, overlay_rgba, src_pts, dst_pts):

  """Proyecta overlay_rgba (con alfa) al plano de base_bgr usando una

  transformación afín definida por src_pts -> dst_pts.

  """

  H, W = base_bgr.shape[:2]

  # Matriz afín

  M = cv2.getAffineTransform(src_pts, dst_pts.astype(np.float32))

  # Warp del PNG y su alfa

  warped = cv2.warpAffine(overlay_rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

  warped_bgr = warped[..., :3]

  warped_alpha = warped[..., 3]



  # Mezcla respetando alfa

  inv_alpha = cv2.bitwise_not(warped_alpha)

  bg_part = cv2.bitwise_and(base_bgr, base_bgr, mask=inv_alpha)

  fg_part = cv2.bitwise_and(warped_bgr, warped_bgr, mask=warped_alpha)

  return cv2.add(bg_part, fg_part)



print("Presiona 'q' para salir.")

while True:

  ret, frame = cap.read()

  if not ret:

    break



  frame = cv2.flip(frame, 1) # espejo

  h, w = frame.shape[:2]



  # Procesar con FaceMesh

  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  res = face_mesh.process(rgb)



  out = np.zeros_like(frame) # por si no detecta, queda negro

  if res.multi_face_landmarks:

    lms = res.multi_face_landmarks[0].landmark



    # Puntos clave para la máscara (ojos + mentón)

    (rx, ry) = (int(lms[IDX_RIGHT_EYE].x * w), int(lms[IDX_RIGHT_EYE].y * h))

    (lx, ly) = (int(lms[IDX_LEFT_EYE ].x * w), int(lms[IDX_LEFT_EYE ].y * h))

    (cx, cy) = (int(lms[IDX_CHIN   ].x * w), int(lms[IDX_CHIN   ].y * h))

    face_pts_affine = np.float32([(rx, ry), (lx, ly), (cx, cy)])



    # Recorte: solo la cara (óvalo)

    face_oval_pts = landmarks_to_points(lms, w, h, FACE_OVAL_IDX)

    face_only, face_mask = keep_only_face(frame, face_oval_pts)



    # Superponer máscara de Guy Fawkes ajustada por afín

    out = overlay_affine(face_only, mask_rgba, mask_pts, face_pts_affine)



    # Opcional: limpiar fuera del óvalo (garantizar fondo negro)

    out = cv2.bitwise_and(out, out, mask=face_mask)



  cv2.imshow("Cara con máscara (fondo negro)", out)



  if cv2.waitKey(1) & 0xFF == ord('q'):

    break



cap.release()

cv2.destroyAllWindows()

