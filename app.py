from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)
stereo_params = {'numDisparities': 16, 'blockSize': 15}

def init_cameras():
    try:
        cam_l = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cam_r = cv2.VideoCapture(1, cv2.CAP_V4L2)
        for cam in (cam_l, cam_r):
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        return cam_l, cam_r
    except:
        return None, None

cam_left, cam_right = init_cameras()

def generate_frames():
    while True:
        if cam_left and cam_right:
            success_l, frame_l = cam_left.read()
            success_r, frame_r = cam_right.read()
        else:
            success_l, success_r = False, False

        if not success_l or not success_r:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, "KAMERA HIBA", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        nd = max(16, (stereo_params['numDisparities'] // 16) * 16)
        bs = max(5, stereo_params['blockSize'])
        if bs % 2 == 0: bs += 1

        stereo = cv2.StereoBM_create(numDisparities=nd, blockSize=bs)
        disparity = stereo.compute(gray_l, gray_r)
        norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_disparity = cv2.applyColorMap(norm_disparity, cv2.COLORMAP_JET)

        ret, buffer = cv2.imencode('.jpg', color_disparity)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    data = request.json
    if 'numDisparities' in data: stereo_params['numDisparities'] = int(data['numDisparities'])
    if 'blockSize' in data: stereo_params['blockSize'] = int(data['blockSize'])
    return jsonify({"status": "success", "params": stereo_params})

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000, threaded=True)
