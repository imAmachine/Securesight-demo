from flask import Flask, Response, render_template, jsonify
from render import ActionDetectionSystem, generate_frames

app = Flask(__name__)
detection_system = ActionDetectionSystem(max_objects=15)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(detection_system), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/objects_data')
def objects_data():
    return jsonify(detection_system.get_current_objects())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)