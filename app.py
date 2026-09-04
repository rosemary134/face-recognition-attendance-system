import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time  

app = Flask(__name__)

nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier('D:/Code/NhanDien/NhanDien/face-recognition-based-attendance-system-master/haarcascade_frontalface_default.xml')


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Lop,Room,Time')  

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    lops = df['Lop']
    rooms = df['Room']
    times = df['Time']
    l = len(df)
    return names, rolls, lops, rooms, times, l

def add_attendance(name, room):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    userclass = name.split('_')[2]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{userclass},{room},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    lops = []
    l = len(userlist)

    for i in userlist:
        name, roll, lop = i.split('_')
        names.append(name)
        rolls.append(roll)
        lops.append(lop)

    return userlist, names, rolls, lops, l

def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser + '/' + i)
    os.rmdir(duser)


# Trang chủ
@app.route('/')
def home():
    names, rolls, lops, rooms, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, lops=lops, rooms=rooms, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Danh sách user
@app.route('/listusers')
def listusers():
    userlist, names, rolls, lops, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, lops=lops, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Xóa user
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/' + duser)

    if os.listdir('static/faces/') == []:
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except:
        pass

    userlist, names, rolls, lops, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, lops=lops, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/start', methods=['GET'])
def start():
    room = request.args.get('room')  
    names, rolls, lops, rooms, times, l = extract_attendance()

    
    if any(room in rooms[i] for i in range(l)):
        return render_template('home.html', names=names, rolls=rolls, lops=lops, rooms=rooms, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='Phòng đã có người mượn, vui lòng chọn phòng khác.')

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, lops=lops, rooms=rooms, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    recognized_names = set() 
    first_recognized_name = None  

    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]

           
            if identified_person not in recognized_names:
                recognized_names.add(identified_person)  
                
                
                if first_recognized_name is None:
                    first_recognized_name = identified_person  
                    add_attendance(first_recognized_name, room)  

            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, lops, rooms, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, lops=lops, rooms=rooms, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    newclass = request.form['newclass']

    userimagefolder = f'static/faces/{newusername}_{newuserid}_{newclass}'

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    i, j = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while 1:
        _, frame = cap.read()

        faces = extract_faces(frame)

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (255, 0, 20),
                2
            )

            cv2.putText(
                frame,
                f'Images Captured: {i}/{nimgs}',
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 20),
                2,
                cv2.LINE_AA
            )

            if j % 5 == 0:

                name = f'{newusername}_{newuserid}_{newclass}_{i}.jpg'

                cv2.imwrite(
                    userimagefolder + '/' + name,
                    frame[y:y+h, x:x+w]
                )

                i += 1

                time.sleep(5)

            j += 1

        if j == nimgs * 5:
            break

        cv2.imshow('Face Capture', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")

    # Lấy lại dữ liệu điểm danh
    names, rolls, lops, rooms, times, l = extract_attendance()

    # Quay lại trang chủ
    return render_template(
        'home.html',
        names=names,
        rolls=rolls,
        lops=lops,
        rooms=rooms,
        times=times,
        l=l,
        totalreg=totalreg(),
        datetoday2=datetoday2
    )
if __name__ == '__main__':
    app.run(debug=True)

