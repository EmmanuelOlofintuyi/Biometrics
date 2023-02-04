# Iris Recognition
# 01. Acquisition-module stub.

import cv2

def acquire_from_file(file_path, view=False):
    iris = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if view:
        cv2.imshow('press any key', iris)
        cv2.waitKey(0)

    print('[INFO] Acquired iris from file', file_path + '.')
    return iris
