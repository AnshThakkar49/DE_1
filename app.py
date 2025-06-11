import os
import base64
import numpy as np
import base64
import cv2
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from report_analysis import analyze_blood_report, analyze_sentiment, generate_html_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = 'hellomotto'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dummy user database
users = {
    "ansh": "ansh@4904",
    "tanisha": "tanisha@123",
    "meet": "meet@456"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('user_dashboard'))
        else:
            mg = "Username or password entered is Incorrect!"
            return render_template('login.html', msg=mg)
    return render_template('login.html')

@app.route('/ud')
def user_dashboard():
    if 'username' in session:
        return render_template('ud.html', username=session['username'])
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/upload_report', methods=['GET', 'POST'])
def upload_report():
    if request.method == 'POST':
        if 'report_file' not in request.files:
            return render_template('report_analyzer.html', msg="No file part")

        file = request.files['report_file']

        if file.filename == '':
            return render_template('report_analyzer.html', msg="No selected file")

        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            results = analyze_blood_report(filepath)
            sentiment = analyze_sentiment(" ".join(results))

            return render_template('report_result.html', results=results, sentiment=sentiment)
        else:
            return render_template('report_analyzer.html', msg="Invalid file format. Only PDFs are allowed.")

    return render_template('report_analyzer.html')

@app.route('/skin_live_camera')
def skin_live_camera():
    return render_template('skin_live_camera.html')


@app.route('/predict_skin_live', methods=['POST'])
def predict_skin_live():
    from PIL import Image
    from io import BytesIO
    import base64
    import os
    import numpy as np
    import cv2
    from tensorflow.keras.models import load_model

    image_data = request.form.get('image_data')
    if not image_data:
        return "No image data", 400

    image_data = image_data.split(',')[1]  # remove the data:image/...;base64,
    img_bytes = base64.b64decode(image_data)
    img = Image.open(BytesIO(img_bytes)).convert('L')

    save_path = os.path.join('static', 'skin_uploads', 'live_capture.jpg')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

    # Model prediction
    model = load_model('model.h5')
    img_cv = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
    img_cv = cv2.resize(img_cv, (128, 128))
    img_cv = img_cv.astype('float32') / 255.0
    img_cv = np.expand_dims(img_cv, axis=-1)
    img_cv = np.expand_dims(img_cv, axis=0)

    prediction = model.predict(img_cv)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    class_names = ['Acne', 'Eczema', 'Psoriasis', 'Healthy']

    if confidence < 0.5:
        result = "Healthy (Low Confidence)"
    else:
        result = f"{class_names[predicted_class]} ({confidence*100:.2f}%)"

    return render_template('skin_live_camera.html', result=result, image_path=save_path)

@app.route('/doctors')
def doctors():
    return render_template('doctors.html')

@app.route('/form')
def form_page():
    return render_template('form.html')

@app.route('/report_analyzer')
def report_analyzer():
    return render_template('report_analyzer.html')

@app.route('/contact')
def contact_us():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)