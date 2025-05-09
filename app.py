from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import urllib.request

app = Flask(__name__)

# Configuración de la URL de streaming
_URL = 'http://192.168.250.169'
_PORT = '81'
_ST = '/stream'
SEP = ':'
stream_url = ''.join([_URL, SEP, _PORT, _ST])

# Parámetros de ruido
gaussian_mean = 0
gaussian_sigma = 25
speckle_variance = 0.1

selected_operation = 'and'

def add_label(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 255, 0)

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (image.shape[1] - text_width) // 2
    y = 20 

    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def apply_noise_and_filters():
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_mean = gaussian_mean
        current_sigma = gaussian_sigma
        current_speckle = speckle_variance

        original = frame.copy()

        # Agregar ruido Gaussiano
        noisy_gauss = np.clip(frame + np.random.normal(current_mean, current_sigma, frame.shape), 0, 255).astype(np.uint8)

        # Agregar ruido Speckle
        speckle = frame + frame * np.random.randn(*frame.shape) * current_speckle
        noisy_speckle = np.clip(speckle, 0, 255).astype(np.uint8)

        # Aplicar filtros
        filtered_median = cv2.medianBlur(noisy_gauss, 5)
        filtered_blur = cv2.blur(noisy_gauss, (5, 5))
        filtered_gaussian = cv2.GaussianBlur(noisy_gauss, (5, 5), 0)

        # Detección de bordes sin filtro (directo sobre ruido)
        canny_raw = cv2.Canny(cv2.cvtColor(noisy_gauss, cv2.COLOR_BGR2GRAY), 100, 200)

        sobel_x_raw = cv2.Sobel(cv2.cvtColor(noisy_gauss, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
        sobel_y_raw = cv2.Sobel(cv2.cvtColor(noisy_gauss, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)
        sobel_raw = cv2.magnitude(sobel_x_raw, sobel_y_raw)
        sobel_raw = cv2.normalize(sobel_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Detección de bordes con filtro (sobre imagen filtrada)
        canny_filtered = cv2.Canny(cv2.cvtColor(filtered_gaussian, cv2.COLOR_BGR2GRAY), 100, 200)

        sobel_x_f = cv2.Sobel(cv2.cvtColor(filtered_gaussian, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
        sobel_y_f = cv2.Sobel(cv2.cvtColor(filtered_gaussian, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)
        sobel_filtered = cv2.magnitude(sobel_x_f, sobel_y_f)
        sobel_filtered = cv2.normalize(sobel_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convertir a 3 canales para combinar
        canny_raw = cv2.cvtColor(canny_raw, cv2.COLOR_GRAY2BGR)
        sobel_raw = cv2.cvtColor(sobel_raw, cv2.COLOR_GRAY2BGR)
        canny_filtered = cv2.cvtColor(canny_filtered, cv2.COLOR_GRAY2BGR)
        sobel_filtered = cv2.cvtColor(sobel_filtered, cv2.COLOR_GRAY2BGR)

        # Etiquetas
        add_label(original, "Original")
        add_label(noisy_gauss, "Ruido Gaussiano")
        add_label(noisy_speckle, "Ruido Speckle")

        add_label(filtered_median, "Filtro Mediana")
        add_label(filtered_blur, "Filtro Blur")
        add_label(filtered_gaussian, "Filtro Gaussiano")

        add_label(canny_raw, "Canny (sin filtro)")
        add_label(canny_filtered, "Canny (con filtro)")
        add_label(sobel_raw, "Sobel (sin filtro)")
        add_label(sobel_filtered, "Sobel (con filtro)")

        # Combinar imágenes en filas
        row1 = np.hstack((original, noisy_gauss, noisy_speckle))
        row2 = np.hstack((filtered_median, filtered_blur, filtered_gaussian))
        row3 = np.hstack((canny_raw, canny_filtered, np.zeros_like(original)))
        row4 = np.hstack((sobel_raw, sobel_filtered, np.zeros_like(original)))

        combined = np.vstack((row1, row2, row3, row4))   

        _, buffer = cv2.imencode('.jpg', combined)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def motion():
    # Camara computadora
    # cap = cv2.VideoCapture(0)

    # Camara esp32
    cap = cv2.VideoCapture(stream_url)
    prev_frame_time = 0
    
    if not cap.isOpened():
        return

    ret, prev_frame = cap.read()
    if not ret:
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        motion_only = apply_bitwise_operation(frame, mask_3ch)

        prev_gray = gray

        _, buffer = cv2.imencode('.jpg', motion_only)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


def apply_bitwise_operation(frame, mask):
    global selected_operation

    if selected_operation == 'and':
        return cv2.bitwise_and(frame, mask)
    elif selected_operation == 'or':
        return cv2.bitwise_or(frame, mask)
    elif selected_operation == 'xor':
        return cv2.bitwise_xor(frame, mask)
    else:
        return cv2.bitwise_and(frame, mask)


def stack_frames(frames):
    resized = [cv2.resize(f, (320, 240)) for f in frames]
    return cv2.hconcat(resized)


def process_and_stack(frame):
    original_color = cv2.resize(frame.copy(), (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hist_eq = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    bilateral_filtered = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)

    img2 = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
    img3 = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    img4 = cv2.cvtColor(bilateral_filtered, cv2.COLOR_GRAY2BGR)

    stacked = stack_frames([original_color, img2, img3, img4])
    return stacked


def video_stream():
    # Camara computadora
    # cap = cv2.VideoCapture(0) 

    # Camara esp32
    cap = cv2.VideoCapture(stream_url)
    prev_frame_time = 0
    
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara.")

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            processed_frame = process_and_stack(frame)

            cv2.putText(processed_frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print(f"Error en el flujo de video: {e}")

    finally:
        cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/motion')
def motion_page():
    return render_template('motion.html')


@app.route('/video_stream')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/motion_stream')
def motion_feed():
    return Response(motion(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_operation', methods=['POST'])
def set_operation():
    global selected_operation
    selected_operation = request.form['operation']
    return render_template('motion.html')


@app.route('/noise_filter')
def noise_filter_page():
    return render_template('noise_filter.html')


@app.route('/noise_filter_stream')
def noise_filter_feed():
    def generate():
        return apply_noise_and_filters()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_noise_params', methods=['POST'])
def set_noise_params():
    global gaussian_mean, gaussian_sigma, speckle_variance
    try:
        gaussian_mean = float(request.form.get('gaussian_mean', 0))
        gaussian_sigma = float(request.form.get('gaussian_sigma', 25))
        speckle_variance = float(request.form.get('speckle_variance', 0.1))
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(debug=True)