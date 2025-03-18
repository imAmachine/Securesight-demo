from flask import Flask, Response, render_template, jsonify, request
from render import ActionDetectionSystem, EmotionDetectionSystem, generate_frames

app = Flask(__name__)

detection_system = ActionDetectionSystem(max_objects=15)
emotion_system = EmotionDetectionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    model_type = request.args.get('model', 'behavior')
    camera_id = int(request.args.get('camera', '0'))
    current_system = detection_system if model_type == 'behavior' else emotion_system
    return Response(
        generate_frames(current_system, camera_id=camera_id), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/objects_data')
def objects_data():
    model_type = request.args.get('model', 'behavior')
    current_system = detection_system if model_type == 'behavior' else emotion_system
    return jsonify(current_system.get_current_objects())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)