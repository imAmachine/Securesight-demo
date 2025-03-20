from flask import Flask, Response, render_template, jsonify, request
from render import ActionDetectionSystem, EmotionDetectionSystem, generate_frames
from googleapiclient.http import MediaFileUpload
from drive.drive import get_drive_service
import os
import tempfile

app = Flask(__name__)

detection_system = ActionDetectionSystem(max_objects=8)
emotion_system = EmotionDetectionSystem()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    model_type = request.args.get('model', 'behavior')
    camera_id = int(request.args.get('camera', '0'))
    vedeo_res = (960, 720)
    
    current_system = detection_system if model_type == 'behavior' else emotion_system
    return Response(
        generate_frames(current_system, camera_id=camera_id, vedeo_res=vedeo_res), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/objects_data')
def objects_data():
    model_type = request.args.get('model', 'behavior')
    current_system = detection_system if model_type == 'behavior' else emotion_system
    return jsonify(current_system.get_current_objects())

@app.route('/upload', methods=['POST'])
def upload_to_drive():
    if 'screenshot' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    temp_path = None
    try:
        file = request.files['screenshot']
        
        # Создание временного файла с гарантированным закрытием
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        # Загрузка на Google Drive
        drive_service = get_drive_service()
        file_metadata = {
            'name': os.path.basename(temp_path),
            'parents': ['1GYiP8bCxfNjLYGacuuoWJGeajs_aoHzD']
        }
        
        media = MediaFileUpload(temp_path, mimetype='image/jpeg')
        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        return jsonify({'success': True, 'file_id': uploaded_file.get('id')})

    except Exception as e:
        return jsonify({'success': False, 'error': 'Произошла ошибка при загрузке'}), 500
        
    finally:
        # Тихая очистка временного файла
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass  # Полное игнорирование ошибок удаления

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)