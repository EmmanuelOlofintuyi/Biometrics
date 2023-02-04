import cv2
import dlib

cam = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('../data/shape_predictor_68_face_landmarks.dat')

# while ESC is not pressed
while True:
    _, frame = cam.read()
    gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_boxes = face_detector(gs_frame)
    for box in face_boxes:
        x1 = box.left()
        y1 = box.top()
        x2 = box.right()
        y2 = box.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for (i, face_box) in enumerate(face_boxes):
        landmarks = landmark_predictor(gs_frame, face_box)
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow('camera', frame)
    if cv2.waitKey(1) % 256 == 27:  # press [ESC] to finish program
        break
