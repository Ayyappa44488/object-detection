
from flask import Flask, request, render_template, Response,redirect,url_for
import os
from process import generate_frames
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        input_path = r"static/uploads/input.mp4"
        file.save(input_path)
        timestamp = int(time.time())
        return render_template('index.html',send_file=input_path, timestamp=timestamp)  

    
@app.route('/video_feed')
def video_feed():
    target_label = request.args.get('object_name', default='')
    input_path = r'static/uploads/input.mp4'  
    return Response(generate_frames(input_path, target_label),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/search', methods=['POST'])
def search():
    object_name = request.form['name']
    timestamp = int(time.time())
    return render_template('index.html', object_name=object_name, timestamp=timestamp)

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
