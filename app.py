from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time

app = Flask(__name__)

# Modelo de substracción adaptativa de fondo
backSub = cv2.createBackgroundSubtractorMOG2()

# Parámetros globales para ruido
gaussian_mean = 0
gaussian_sigma = 25
speckle_variance = 0.1

def add_gaussian_noise(image, mean, sigma):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = image + gauss
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_speckle_noise(image, variance):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    noisy_image = image + image * gauss * variance
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def adjust_gamma(image, gamma=1.0):
    # Construye una tabla de mapeo para realizar la corrección gamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Aplica la corrección gamma usando la tabla de mapeo
    return cv2.LUT(image, table)

def video_capture():
    cap = cv2.VideoCapture(0)   
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    prev_frame_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame.")
                break

            # Aplicar ruido gaussiano
            frame = add_gaussian_noise(frame, gaussian_mean, gaussian_sigma)

            # Aplicar ruido speckle
            frame = add_speckle_noise(frame, speckle_variance)

            # Calcular FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # Aplicar substracción de fondo
            fgMask = backSub.apply(frame)

            # Mejorar iluminación con CLAHE
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray)

            # Generar máscara de movimiento
            _, mask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            # Ajustar exposición con Gamma Correction
            gamma_corrected = adjust_gamma(frame, gamma=1.5)

            # Agregar texto (FPS)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(clahe_img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(gamma_corrected, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Combinar resultados en una sola imagen
            combined = np.hstack((
                frame,
                cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR),
                result,
                gamma_corrected
            ))

            # Codificar el frame para transmisión
            (flag, encodedImage) = cv2.imencode(".jpg", combined)
            if not flag:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    finally:
        cap.release()   
        print("Cámara liberada.")

@app.route("/")
def index():
    print("Cargando página principal...")
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    print("Iniciando transmisión de vídeo...")
    return Response(video_capture(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/set_noise_params", methods=["POST"])
def set_noise_params():
    global gaussian_mean, gaussian_sigma, speckle_variance
    gaussian_mean = float(request.form.get("gaussian_mean", 0))
    gaussian_sigma = float(request.form.get("gaussian_sigma", 25))
    speckle_variance = float(request.form.get("speckle_variance", 0.1))
    print("Parámetros de ruido actualizados.")
    return jsonify({"message": "Noise parameters updated successfully."})

if __name__ == "__main__":
    print("Iniciando servidor Flask...")
    app.run(debug=False)