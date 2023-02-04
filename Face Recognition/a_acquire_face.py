import cv2

def acquire_from_file(file_path, view=False):
    image = cv2.imread(file_path)

    if view:
        cv2.imshow('Press any key.', image)
        cv2.waitKey(0)

    print('[INFO] Acquired image from file.')
    return image
