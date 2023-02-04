import cv2
import math
import numpy
import scipy.ndimage


FRAME_WIDTH = 640
FACE_SIZE = 256
VJ_FACE_DETECTOR = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
VJ_EYES_DETECTOR = cv2.CascadeClassifier('./data/haarcascade_eye.xml')

def __rotate_and_crop(image, rad_angle):
    h, w = image.shape

    degree_angle = 360.0 - (180.0 * rad_angle / numpy.pi)
    rotated = scipy.ndimage.rotate(image, degree_angle, reshape=False)

    crop_size = int(round(h / numpy.sqrt(2)))
    crop_start = int((h - crop_size) / 2.0)

    rotated = rotated[crop_start: crop_start + crop_size, crop_start: crop_start + crop_size]
    return rotated


def _01_preprocess(frame, output_width, view=False):
    if len(frame.shape) > 2 and frame.shape[2] > 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aspect_ratio = float(frame.shape[1]) / frame.shape[0]
    height = int(round(output_width / aspect_ratio))
    frame = cv2.resize(frame, (output_width, height))

    if view:
        cv2.imshow('Preprocessing, press any key.', frame)
        cv2.waitKey(0)

    print('[INFO] Preprocessed frame.')
    return frame


def _02_detect_face(grayscale_frame, view=False):
    face_boxes = VJ_FACE_DETECTOR.detectMultiScale(grayscale_frame)

    if len(face_boxes) == 0:
        return None

    x, y, w, h = face_boxes[0]
    face = grayscale_frame[y:y + h, x:x + w]

    if view:
        cv2.imshow('Detected face, press any key.', face)
        cv2.waitKey(0)

    print('[INFO] Detected one face.')
    return face


def _03_align_face(face, view=False):
    eye_boxes = VJ_EYES_DETECTOR.detectMultiScale(face)

    if len(eye_boxes) != 2:
        return None

    x1, y1, w1, h1, = eye_boxes[0]  # eye 1
    x2, y2, w2, h2, = eye_boxes[1]  # eye 2

    if x1 < x2:
        xc1 = x1 + w1 / 2.0  # right eye, mirrored on the left
        yc1 = y1 + h1 / 2.0

        xc2 = x2 + w2 / 2.0  # left eye, mirrored on the right
        yc2 = y2 + h2 / 2.0

    else:
        xc2 = x1 + w1 / 2.0  # left eye, mirrored on the right
        yc2 = y1 + h1 / 2.0

        xc1 = x2 + w2 / 2.0  # right eye, mirrored on the right
        yc1 = y2 + h2 / 2.0

    angle = math.atan2(yc2 - yc1, xc2 - xc1)

    face = __rotate_and_crop(face, -angle)

    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))

    if view:
        cv2.imshow('Aligned face, press any key.', face)
        cv2.waitKey(0)

    print('[INFO] Aligned the face.')
    return face


def _04_fix_illumination(face, view=False):
    face = cv2.equalizeHist(face, face)

    if view:
        cv2.imshow('[INFO] Equalized face, press any key.', face)
        cv2.waitKey(0)

    return face


def enhance(frame, view=False):
    pp_frame = _01_preprocess(frame, FRAME_WIDTH, view=view)

    face = _02_detect_face(pp_frame, view=view)

    if face is not None:
        face = _03_align_face(face, view=view)

    if face is not None:
        face = _04_fix_illumination(face, view=view)

    return face
