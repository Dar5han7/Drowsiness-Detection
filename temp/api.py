from flask import Flask, render_template, request,jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from detection import eye_mouth_crop,cnn
import socket
import select
# import imutils
# from flask_ngrok import run_with_ngrok
app = Flask(__name__)
# run_with_ngrok(app)

HOST = '127.0.0.1'
PORT = 5000
MAX = 100000
connected_clients_sockets = []

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(10)
connected_clients_sockets.append(server_socket)


mp_face_mesh = mp.solutions.face_mesh
model_path = r'../models/model (7).h5'
final = tf.keras.models.load_model(model_path)
camera=cv2.VideoCapture(0)



# @app.route('/', methods = ['GET', 'POST'])
# def upload_file():
count =0
group = []
intermediate_model = cnn()

while True:
    read_sockets, write_sockets, error_sockets = select.select(connected_clients_sockets, [], [])




    for sock in read_sockets:
        # print(sock)

        if sock == server_socket:

            sockfd, client_address = server_socket.accept()
            connected_clients_sockets.append(sockfd)

        else:
            # try:
            data = sock.recv(MAX)
            # print(data)


            npimg = np.frombuffer(data, np.uint8)
            try:
                img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

            except:
                pass

            # frame = imutils.resize(img, width=450)
            # image = frame
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            status = None
            try:
                # with mp_face_mesh.FaceMesh() as face_mesh:
                #   results = face_mesh.process(image)
                #   for face_landmarks in results.multi_face_landmarks:
                #       landmark_arr = []
                #       for point in face_landmarks.landmark:
                #           landmark_arr.append([point.x, point.y, point.z])
                #       # print(landmark_arr)
                #       status = model.predict([landmark_arr])

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
                    cv2.imwrite("../frame1.jpg", image)

                    # print(results.multi_face_landmarks)
                    if results.multi_face_landmarks:
                        # print("1")
                        for face_landmarks in results.multi_face_landmarks:
                            # crop the image#
                            normal_image = eye_mouth_crop(face_landmarks, image)
                            # print(group)
                            # print(normal_image.shape)
                            print(count)
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
                                # print(out)
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
                                msg = label[index]
                                msg = bytes(msg, 'utf-8')

                                # print(msg,"hhhhhhhhhhhhhhhhhhhhhhhhhhhh")
                                sock.send(msg)
            except Exception as e:
                print(e)
                msg = "no output"
                msg = bytes(msg, 'utf-8')

                print(msg)
                sock.send(msg)
            # print(msg)
            # msg = 'alert'
            # if status is not None:
            #     index = np.argmax(status[0])
            #     percentage = status[0][index]
            #     label = ['alert', 'drowsy', 'semi alert']
            #     msg = label[index] + ' with confidence ' + '{:.2f}'.format(percentage)
            # cv2.imwrite('image.png',img)

            #     # sock.shutdown()
            # except Exception as e :
            #     print("stop",e)
            #     sock.close()
            #     connected_clients_sockets.remove(sock)
            #     continue



            # else:
            #     return ''
















































    #
    # print(request.method)
    # if request.method == 'POST':
    #   # while True:
    #   #
    #   #     ## read the camera frame
    #   #     success, frame = camera.read()
    #   #     if not success:
    #   #         break
    #   #     else:
    #   #         ret, buffer = cv2.imencode('.jpg', frame)
    #   #         frame = buffer.tobytes()
    #   #
    #   #     # yield (b'--frame\r\n'
    #       #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    #
    #   f = request.files['file'].read()
    #   count = request.files['count'].read()
    #   count = np.frombuffer(count, np.int8)
    #
    #   print(count)
    #
    #   # print(f)
    #   npimg = np.frombuffer(f, np.uint8)
    #   img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    #
    #   # frame = imutils.resize(img, width=450)
    #   # image = frame
    #   image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    #   # To improve performance, optionally mark the image as not writeable to
    #   # pass by reference.
    #   image.flags.writeable = False
    #   status = None
    #   try:
    #       # with mp_face_mesh.FaceMesh() as face_mesh:
    #       #   results = face_mesh.process(image)
    #       #   for face_landmarks in results.multi_face_landmarks:
    #       #       landmark_arr = []
    #       #       for point in face_landmarks.landmark:
    #       #           landmark_arr.append([point.x, point.y, point.z])
    #       #       # print(landmark_arr)
    #       #       status = model.predict([landmark_arr])
    #       intermediate_model = cnn()
    #
    #       with mp_face_mesh.FaceMesh(
    #               min_detection_confidence=0.5,
    #               min_tracking_confidence=0.5) as face_mesh:
    #           image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    #           # To improve performance, optionally mark the image as not writeable to
    #           # pass by reference.
    #           image.flags.writeable = False
    #           results = face_mesh.process(image)
    #
    #           # Draw the face mesh annotations on the image.
    #           image.flags.writeable = True
    #           image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #           shape = image.shape
    #           if results.multi_face_landmarks:
    #               for face_landmarks in results.multi_face_landmarks:
    #                   # crop the image#
    #                   normal_image = eye_mouth_crop(face_landmarks, image)
    #                   print(group)
    #                   # print(normal_image.shape)
    #
    #                   if count % 50 != 0 or count == 0:
    #                       try:
    #                           intermediate_prediction = intermediate_model.predict(normal_image)
    #                           group.append(intermediate_prediction)
    #                       except:
    #                           pass
    #
    #                       count = count + 1
    #
    #
    #                   else:
    #                       array = np.concatenate(group)
    #                       group = []
    #                       try:
    #                           intermediate_prediction = intermediate_model.predict(normal_image)
    #                           group.append(intermediate_prediction)
    #                       except:
    #                           pass
    #
    #                       X = np.ravel(array)
    #                       X = X.reshape(1, 50, -1)
    #                       out = final.predict(X)
    #                       print(out)
    #                       index = np.argmax(out[0])
    #                       percentage = out[0][index]
    #                       count = count + 1
    #                   if count > 50:
    #                       frame = image
    #                       label = ['alert', 'drowsy']
    #                       # msg = f'{str(label[index])}:{str(round(percentage, 2))}, fps:{str(fps)}'
    #                       # cv2.putText(frame, msg, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
    #                       #             cv2.LINE_AA)
    #                       # mp_drawing.draw_landmarks(
    #                       #     image=image,
    #                       #     landmark_list=face_landmarks,
    #                       #     connections=mp_face_mesh.FACE_CONNECTIONS,
    #                       #     landmark_drawing_spec=drawing_spec,
    #                       #     connection_drawing_spec=drawing_spec)
    #                       msg = label[index]
    #                       print(msg)
    #   except Exception as e:
    #       print(e)
    #       msg="no output"
    #       print(msg)
    #   # print(msg)
    #   # msg = 'alert'
    #   # if status is not None:
    #   #     index = np.argmax(status[0])
    #   #     percentage = status[0][index]
    #   #     label = ['alert', 'drowsy', 'semi alert']
    #   #     msg = label[index] + ' with confidence ' + '{:.2f}'.format(percentage)
    #   # cv2.imwrite('image.png',img)
    #   return jsonify({'result': msg })
    # else:
    #   return ''
# if __name__ == '__main__':
#     app.run(host='127.0.0.1', debug=True,port="5000")

    # app.run()