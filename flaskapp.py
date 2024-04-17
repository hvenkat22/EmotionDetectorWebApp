import cv2
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/afterclick',methods=['GET','POST'])
def afterclick():
    img=request.files['file']
    img.save('static/imgfile.jpg')

    img1 = cv2.imread('static/imagfile.jpg')
    grey=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces=cascade.detectMultiScale(grey,1.1,3)

    for x,y,w,h in faces:
        cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)

        crop = img1[y:y+h,x:x+w]
    
    cv2.imwrite('static/after.jpg',img1)

    try:
        cv2.imwrite('static/croppedimg.jpg',crop)
    except:
        pass

    try:
        image = cv2.imread('static/croppedimg.jpg',0)
    except:
        image = cv2.imread('static/imgfile.jpg',0)

    image = cv2.resize(image, (48,48))
    image = image/255.0
    image = np.reshape(image,(1,48,48,1))

    model=load_model('EmotionDetectionModel.h5') #EmotionDetectionModel.ipynb

    pred = model.predict(image)
    labels=['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprised']

    pred=np.argmax(pred)
    final=labels[pred]

    return render_template('afterclick.html',data=final)

if __name__ == "__main__":
    app.run(debug=True)
