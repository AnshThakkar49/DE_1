import flask
import os
from flask import Flask, render_template, request, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from report_analysis import analyze_blood_report, analyze_sentiment

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)

app.secret_key='hellomotto'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

users={
    "ansh": "ansh@4904",
    "tanisha":"tanisha@123",
    "meet": "meet@456"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in {'pdf'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username]==password:
            session['username'] = username
            return redirect(url_for('user_dashboard'))
            #resp = redirect(url_for('ud'))
            #resp.set_cookies('loggedin_user',username)
            #return resp
        else:
            mg = "Uername or password entered is Incorrect!"
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
    session.pop('username',None)
    #resp = redirect(url_for('login'))
    #resp.set_cookie('loggedin_user','', expires = 0)
    return redirect(url_for('home'))

@app.route('/upload_report', methods=['GET', 'POST'])
def upload_report():
    if request.method == 'POST':
        if 'report_file' not in request.files:
            return render_template('ud.html', msg="No file part")
        file = request.files['report_file']
        if file.filename == '':
            return render_template('ud.html', msg="No selected file")
        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)
            
            results = analyze_blood_report(filepath)
            sentiment = analyze_sentiment(" ".join(results))
            
            return render_template('report_result.html', 
                                   results=results, 
                                   sentiment=sentiment)
    return redirect(url_for('user_dashboard'))


@app.route('/doctors')
def doctors():
    return render_template('doctors.html')

@app.route('/form')
def form_page():
    return render_template('form.html')

@app.route('/contact')
def contact_us():
    return render_template('contact.html')

@app.route('/report')
def report():
    return render_template('report.html')

if __name__ == '__main__':
    app.run(debug=True)