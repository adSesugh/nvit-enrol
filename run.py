import inspect
import cv2
import os
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# App Initialization
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
import os
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('PRODUCTION_DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
app.app_context().push()

# Create a database instance
db = SQLAlchemy(app)
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

nimgs = 10


class Student(db.Model):
    __tablename__ = 'students'

    id = db.Column(db.Integer, primary_key=True, nullable=False, unique=True)
    created = db.Column(db.DateTime(timezone=True),
                        default=datetime.now())
    updated = db.Column(db.DateTime(timezone=True), default=datetime.now(),
                        onupdate=datetime.now())

    student_no = db.Column(db.String(60), unique=True, nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    middle_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    phone_number = db.Column(db.String(20), unique=True)
    gender = db.Column(db.String(15))
    state_of_origin = db.Column(db.String(70))
    lga_of_origin = db.Column(db.String(100))
    dob = db.Column(db.Date)
    course_of_study = db.Column(db.String(190))
    address = db.Column(db.String(190))
    means_of_id = db.Column(db.String(100))
    means_of_id_upload = db.Column(db.String(191), nullable=True)
    employment_status = db.Column(db.String(100))
    level_of_education = db.Column(db.String(100))
    disabled = db.Column(db.String(5))
    disability_detail = db.Column(db.String(1000), nullable=True)

    emergency_full_name = db.Column(db.String(150), nullable=False)
    emergency_email = db.Column(db.String(100), nullable=False)
    emergency_phone_number = db.Column(db.String(20))
    emergency_address = db.Column(db.String(150), nullable=False)

    guardian_first_name = db.Column(db.String(60), nullable=False)
    guardian_last_name = db.Column(db.String(60), nullable=False)
    guardian_email = db.Column(db.String(100), nullable=False)
    guardian_phone_number = db.Column(db.String(20))
    terms = db.Column(db.String(10), default=True, nullable=True)

    def __repr__(self):
        return "<%r> <%r>" % (self.first_name, self.last_name)


# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('StudentNo,FirstName,MiddleName,LastName,Course,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        print(user)
        for imgname in os.listdir(f'static/faces/{user}'):
            print(imgname)
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            print(f'imaf {img}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    mats = df['StudentNo']
    names = f'{df["LastName"]} {df["FirstName"]}'
    courses = df['Course']
    times = df['Time']

    l = len(df)
    return mats, names, courses, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    numbers = []
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        roll, name = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser + '/' + i)
    os.rmdir(duser)


################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    mats, names, courses, times, l = extract_attendance()
    return render_template('home.html', mats=mats, names=names, courses=courses, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


@app.route('/attendance')
def attendance_list():
    mats, names, courses, times, l = extract_attendance()
    return render_template('attendance.html', mats=mats,  names=names, courses=courses, times=times, l=l, totalreg=totalreg)


## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/' + duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/') == []:
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    numbers, names, courses, times, l = extract_attendance()
    return render_template('home.html', numbers=numbers, names=names, courses=courses, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':

        data = request.form.to_dict()

        if 'means_of_id' in request.files:
            means_of_id = request.files['means_of_id_upload']
            if means_of_id:
                photo_path = os.path.join('uploads', means_of_id.filename)
                means_of_id.save(photo_path)
                data['means_of_id'] = means_of_id.filename

        year = abs(datetime.now().year) % 100

        data['student_no'] = 'NVITwb/' + str(year) + '/' + str(Student.query.count() + 1).zfill(6)
        data['terms'] = 'Yes' if request.form['terms'] else 'No'

        first_name = data['first_name']
        mat = str(Student.query.count() + 1).zfill(6)

        student = Student(**data)
        db.session.add(student)
        db.session.commit()

        userimagefolder = 'static/faces/' + first_name + '_' + str(mat)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = first_name + '_' + str(i) + '.jpg'
                    cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            if j == nimgs * 5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        mats, names, courses, times, l = extract_attendance()
        return render_template('home.html', mats=mats, names=names, courses=courses, times=times, l=l, totalreg=totalreg())

    return render_template('home.html')


if __name__ == "__main__":
    db.create_all()
    app.run(host='0.0.0.0', port=5000)