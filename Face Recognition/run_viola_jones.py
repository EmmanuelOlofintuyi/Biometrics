import cv2

cam = cv2.VideoCapture(0)
vj_face_detector = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
vj_eyes_detector = cv2.CascadeClassifier('../data/haarcascade_eye.xml')

# while ESC is not pressed
while True:
    _, frame = cam.read()
    gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_boxes = vj_face_detector.detectMultiScale(gs_frame)
    for box in face_boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for box in face_boxes:
        x_face, y_face, w_face, h_face = box
        face = gs_frame[y_face:y_face + h_face, x_face:x_face + w_face]

        eye_boxes = vj_eyes_detector.detectMultiScale(face)
        for eye_box in eye_boxes:
            x_eye, y_eye, w_eye, h_eye = eye_box
            radius = int(round(max(w_eye / 2.0, h_eye / 2.0)))
            cv2.circle(frame, (x_face + x_eye + radius, y_face + y_eye + radius), radius, (0, 0, 255), 2)

    cv2.imshow('camera', frame)
    if cv2.waitKey(1) % 256 == 27:  # press [ESC] to finish program
        break
