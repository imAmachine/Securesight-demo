from flask import Flask, Response, request, jsonify, render_template
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

@app.route('/set_method', methods=['POST'])
def set_method():
    data = request.get_json()
    method = data.get('method', 'action')
    
    success = detection_system.set_processing_method(method)
    if success:
        return jsonify({"status": "success", "method": method})
    else:
        return jsonify({"status": "error", "message": f"Method '{method}' not found"}), 400

@app.route('/get_available_methods', methods=['GET'])
def get_available_methods():
    methods = list(detection_system.processing_methods.keys())
    return jsonify({"methods": methods})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)