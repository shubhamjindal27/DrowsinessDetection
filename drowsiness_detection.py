import cv2
import os
from keras.models import load_model
import numpy as np
#from pygame import mixer
import time
from flask import Flask, render_template, Response, redirect

app = Flask(__name__)

#mixer.init()
#sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

model = load_model('app/models/cnncat2.h5')
#path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

camera = None
display_video = False

sound_playing = False

def generate_frames():
    global camera, sound, sound_playing, display_video
    #mixer.init()
    score = 0
    rpred = [99]
    lpred = [99]
    thicc = 2
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            height,width = frame.shape[:2] 
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
            left_eye = leye.detectMultiScale(gray)
            right_eye =  reye.detectMultiScale(gray)
            
            cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 2)

                for (x,y,w,h) in right_eye:
                    r_eye=frame[y:y+h,x:x+w]
                    r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                    r_eye = cv2.resize(r_eye,(24,24))
                    r_eye= r_eye/255
                    r_eye=  r_eye.reshape(24,24,-1)
                    r_eye = np.expand_dims(r_eye,axis=0)
                    rpred = model.predict(r_eye)
                    rpred = np.argmax(rpred, axis=1)
                    break

                for (x,y,w,h) in left_eye:
                    l_eye=frame[y:y+h,x:x+w]
                    l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
                    l_eye = cv2.resize(l_eye,(24,24))
                    l_eye= l_eye/255
                    l_eye=l_eye.reshape(24,24,-1)
                    l_eye = np.expand_dims(l_eye,axis=0)
                    lpred = model.predict(l_eye)
                    lpred = np.argmax(lpred, axis=1)
                    break

                break
            
            if(rpred[0]==0 and lpred[0]==0):
                score=score+1
                cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            else:
                if(score > 0):
                    score=score-1
                cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

            cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

            if(score>6):
                #person is feeling sleepy so we beep the alarm
                #cv2.imwrite(os.path.join(path,'image.jpg'),frame)
                if not sound_playing:
                    try:
                        #sound.play(loops=-1)
                        sound_playing = True
                    except:  # isplaying = False
                        pass

                if(thicc<16):
                    thicc = thicc+2
                else:
                    if thicc>2:
                        thicc = thicc-2
                cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
            else:
                #sound.stop()
                sound_playing = False
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
        yield(b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', display_video=display_video)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global camera, display_video, sound
    if camera is None:
        camera = cv2.VideoCapture(0)
        display_video = True
    return redirect('/')

@app.route('/stop')
def stop():
    global camera, display_video, sound, sound_playing
    if camera is not None:
        camera.release()
        camera = None
        display_video = False
        #mixer.quit()
        sound_playing = False
    return redirect('/')

if __name__ == "__main__":
    app.run()
