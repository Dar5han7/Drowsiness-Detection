import cv2
import mediapipe as mp
import os
import shutil



mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def video_input(pathor0,name ="driver",folder="alert"):

    cap = cv2.VideoCapture(pathor0)

    newpath_eye = f"photos/{name}/{folder}/"


    if folder != None:
        if os.path.exists(newpath_eye):
            shutil.rmtree(newpath_eye)
            os.makedirs(newpath_eye)
        else:
            os.makedirs(newpath_eye)

    count = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            cv2.waitKey(10)

            if not success:
              print("Ignoring empty camera frame.")
              break

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(count,fps)

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
                    eye,count = eye_mouth_crop(face_landmarks,image,newpath_eye,name,count)

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

def eye_mouth_crop(face_landmarks,image,newpath_eye,name,count=0):
    "function to crop the eye and mouth from the images"

    left_eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
    right_eye = [263,466,388,387,386,385,384,368,362,382,381,380,374,373,390,249]
    # mouth = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

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

    if newpath_eye !=None:

      output_name1 = newpath_eye + name + str(count) + ".jpg"

      try:
        croppedImg1 = cv2.resize(croppedImg1, (150, 75))

        cv2.imwrite(output_name1, croppedImg1)
      except:
          pass
      count = count + 1
    else:
      pass

    return croppedImg1,count


if __name__ == "__main__":
    # extract = video_input(r"10.mp4","darr","alert")# webcam
    extract = video_input(0,"darr","alert")# webcam

    # print(extract)
