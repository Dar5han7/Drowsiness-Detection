from flask import Flask, render_template, Response, jsonify
from api_testing import VideoCamera
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from detection import eye_mouth_crop,cnn


app = Flask(__name__)

video_stream = VideoCamera()

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
model_path = r'../models/model (7).h5'
final = tf.keras.models.load_model(model_path)


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        intermediate_model = cnn()
        group = []
        count = 0

        image,frame = camera.get_frame()
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
                    # print(normal_image.shape)

                    if count % 50 != 0 or count == 0:
                        try:
                            intermediate_prediction = intermediate_model.predict(normal_image)
                            group.append(intermediate_prediction)
                        except:
                            pass

                        count = count + 1


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
                        out = final.predict(X)
                        print(out)
                        index = np.argmax(out[0])
                        percentage = out[0][index]
                        count = count + 1
                    if count > 50:
                        frame = image
                        label = ['alert', 'drowsy']
                        # msg = f'{str(label[index])}:{str(round(percentage, 2))}, fps:{str(fps)}'
                        # cv2.putText(frame, msg, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        #             cv2.LINE_AA)
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACE_CONNECTIONS,
                    #     landmark_drawing_spec=drawing_spec,
                    #     connection_drawing_spec=drawing_spec)
                        return label[index]

            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    status = gen(video_stream)

    return jsonify({'result': status})

    # return Response(gen(video_stream),
    #             mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # app.run(host='127.0.0.1', debug=True,port="5000")
    app.run()