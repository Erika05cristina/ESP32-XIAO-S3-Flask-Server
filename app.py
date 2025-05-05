from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara.")
    
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        processed_frame = process_and_stack(frame)

        # FPS en la imagen final
        cv2.putText(processed_frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_stream')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
