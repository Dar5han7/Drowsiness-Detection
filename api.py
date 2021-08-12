from flask import Flask, render_template, request,jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from detection import eye_mouth_crop,cnn

app = Flask(__name__)



mp_face_mesh = mp.solutions.face_mesh
model_path = r'models/model (7).h5'
final = tf.keras.models.load_model(model_path)
camera=cv2.VideoCapture(0)

intermediate_model = cnn()
group = []


@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    global group

    if request.method == 'POST':
        f = request.files['file'].read()
        npimg = np.frombuffer(f, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        status = None
        try:
            with mp_face_mesh.FaceMesh(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                shape = image.shape

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # crop the image#
                        normal_image = eye_mouth_crop(face_landmarks, image)

                        if (len(group)) % 50 != 0 or (len(group)) == 0:

                            try:
                                intermediate_prediction = intermediate_model.predict(normal_image)
                                group.append(intermediate_prediction)
                            except:
                                pass
                            msg = "no output wait till 50 frames"

                        else:
                            array = np.concatenate(group)
                            group = []
                            try:
                                intermediate_prediction = intermediate_model.predict(normal_image)
                                group.append(intermediate_prediction)
                            except:
                                pass

                            X = np.ravel(array)
                            X = X.reshape(1, 50, -1)
                            # print(X.shape)
                            out = final.predict(X)
                            # print(out)
                            index = np.argmax(out[0])
                            percentage = out[0][index]
                            frame = image
                            label = ['alert', 'drowsy']

                            msg = label[index]
                else:
                    msg = "face not captured"


        except:
            msg="no output"

    else:
        msg = "no output"
    print(msg,"hhh")
    return jsonify({'result': msg})


if __name__ == '__main__':
    # app.run(host='127.0.0.1', debug=True,port="5000")

    app.run()







