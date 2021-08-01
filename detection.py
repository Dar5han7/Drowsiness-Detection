import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np



mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def video_input(pathor0):
    model_path = r'models/model (7).h5'
    final = tf.keras.models.load_model(model_path)

    intermediate_model =cnn()

    cap = cv2.VideoCapture(pathor0)


    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        group= []
        count =0
        while cap.isOpened():
            success, image = cap.read()

            if not success:
              print("Ignoring empty camera frame.")
              # If loading a video, use 'break' instead of 'continue'.
              break


            fps = int(cap.get(cv2.CAP_PROP_FPS))

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
                    #crop the image#
                    normal_image = eye_mouth_crop(face_landmarks,image)
                    # print(normal_image.shape)

                    if count % 50 != 0 or count==0:
                        try:
                            intermediate_prediction = intermediate_model.predict(normal_image)
                            group.append(intermediate_prediction)
                        except:
                            pass

                        count = count +1


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
                        count = count +1
                    if count>50:
                        frame = image
                        label = ['alert', 'drowsy']
                        msg = f'{str(label[index])}:{str(round(percentage,2))}, fps:{str(fps)}'
                        cv2.putText(frame, msg, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                    cv2.LINE_AA)
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACE_CONNECTIONS,
                    #     landmark_drawing_spec=drawing_spec,
                    #     connection_drawing_spec=drawing_spec)

                    cv2.imshow('MediaPipe FaceMesh', image)
                # cv2.waitKey(1000)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            else:
                pass
    cap.release()

def eye_mouth_crop(face_landmarks,image):

    "function to crop the eye and mouth from the images"

    left_eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
    right_eye = [263,466,388,387,386,385,384,368,362,382,381,380,374,373,390,249]
    mouth = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

    x1= face_landmarks.landmark[left_eye[0]].x
    y1= face_landmarks.landmark[left_eye[0]].y

    x2 = face_landmarks.landmark[right_eye[0]].x
    y2 = face_landmarks.landmark[right_eye[0]].y

    shape = image.shape

    relative_x1 = int(x1 * shape[1])
    relative_y1 = int(y1 * shape[0])
    relative_x2 = int(x2 * shape[1])
    relative_y2 = int(y2 * shape[0])


    scale = int(shape[0]/50)


    croppedImg1 = image[min(relative_y1,relative_y2) - scale:max(relative_y1,relative_y2) + scale, relative_x1 - scale:relative_x2 + scale]

    try:
        croppedImg1 = cv2.resize(croppedImg1, (150, 75))
    except:
        pass

    croppedImg1 = croppedImg1/255
    test_image = np.expand_dims(croppedImg1, axis=0)


    return test_image

def cnn():


    IMAGE_SIZE = [75, 150]

    # model_path = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    cnn = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False, input_tensor=None, weights="imagenet",
        input_shape=IMAGE_SIZE + [3], pooling=None, classes=2,
        classifier_activation='softmax',)
    for layer in cnn.layers:
        layer.trainable = False
    layer_output = cnn.get_layer("mixed10").output
    intermediate_model = tf.keras.models.Model(inputs=cnn.input, outputs=layer_output)

    return intermediate_model









if __name__ == "__main__":

    # video_input(r"D:\images\10.mp4")# file

    video_input(0)# webcam


