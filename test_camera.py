import cv2

#  abrir la cámara
cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
else:
    print("Cámara abierta correctamente.")

# Muestra el vídeo capturado
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    cv2.imshow("Test Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()