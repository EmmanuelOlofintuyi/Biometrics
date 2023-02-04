# Iris Recognition
# 02. Module to enhance iris samples, aiming at further feature extraction.

import math
import numpy
import cv2
from scipy.ndimage import gaussian_filter1d

IRIS_WIDTH = 640
IRIS_BLUR_SIZE = 31
MIN_PUPIL_R = 20
MAX_PUPIL_R = 50
MIN_PUPIL_X = 220
MAX_PUPIL_X = 420
MIN_PUPIL_Y = 140
MAX_PUPIL_Y = 340
MIN_IRIS_R = 100
MAX_IRIS_R = 200
PIXEL_STEP = 5
SMOOTH_SIGMA = 1.2
NORM_IRIS_SECTOR_COUNT = 1000
NORM_IRIS_WIDTH = 512
NORM_IRIS_HEIGHT = 64


def _circular_integrate(iris, x0, y0, r, sampling=360.0):
    rows, cols = iris.shape

    pixel_values = []

    for angle in numpy.arange(0, 2.0 * numpy.pi, 2.0 * numpy.pi / sampling):
        x = int(round(x0 + r * math.sin(angle)))
        y = int(round(y0 + r * math.cos(angle)))

        if x < 0 or x >= cols or y < 0 or y >= rows:
            return None

        pixel_values.append(float(iris[y, x]))

    return numpy.array(pixel_values)


def _circular_differentiate(iris, x0, y0, r):
    values = []
    for current_r in [r, r + 1]:
        v = _circular_integrate(iris, x0, y0, current_r)
        if v is not None:
            values.append(v)

    if len(values) < 2:
        return None

    return numpy.mean(abs(values[0] - values[1]))


def _fit_circumference(iris, min_r, max_r, min_x, max_x, min_y, max_y, pixel_step, smooth_sigma):
    r_values = range(min_r, max_r, pixel_step)

    circs = []
    for x0 in range(min_x, max_x, pixel_step):
        for y0 in range(min_y, max_y, pixel_step):
            v_r = []
            circs_r = []

            for r in r_values:
                v = _circular_differentiate(iris, x0, y0, r)
                if v is not None:
                    v_r.append(v)
                    circs_r.append(r)

            if len(v_r) > 0:
                v_r = gaussian_filter1d(v_r, sigma=smooth_sigma)

                max_ind = numpy.argmax(v_r)
                circs.append((v_r[max_ind], x0, y0, circs_r[max_ind]))

    if len(circs) == 0:
        return None

    best_c = sorted(circs, key=lambda x: (x[0], x[3]), reverse=True)[0]
    return best_c[1], best_c[2], best_c[3]  # x position, y position, radius


def _01_preprocess(iris, output_width, view=False):
    if len(iris.shape) > 2 and iris.shape[2] > 1:  # more than one channel?
        iris = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)

    aspect_ratio = float(iris.shape[0]) / iris.shape[1]
    height = int(round(output_width * aspect_ratio))
    iris = cv2.resize(iris, (output_width, height))

    if view:
        cv2.imshow('Preprocessing, press any key.', iris)
        cv2.waitKey(0)

    print('[INFO] Preprocessed iris.')
    return iris


def _02_mask_pupil_and_specular_reflections(iris, blur_size, view=False):
    blurred_iris = cv2.medianBlur(iris, blur_size)
    blurred_iris = cv2.equalizeHist(blurred_iris)
    if view:
        cv2.imshow('Blurred iris, press any key.', blurred_iris)
        cv2.waitKey(0)

    _, pupil_mask = cv2.threshold(blurred_iris, 1, 255, cv2.THRESH_BINARY)
    if view:
        cv2.imshow('Pupil mask, press any key.', pupil_mask)
        cv2.waitKey(0)

    iris = cv2.equalizeHist(iris)
    _, specr_mask = cv2.threshold(iris, 254, 255, cv2.THRESH_BINARY_INV)
    if view:
        cv2.imshow('Specular reflection mask, press any key.', specr_mask)
        cv2.waitKey(0)

    print('[INFO] Computed pupil and specular reflection masks.')
    return blurred_iris, pupil_mask, specr_mask


def _03_detect_iris_boundaries(blurred_iris, pupil_mask,
                               min_pupil_r, max_pupil_r,
                               min_pupil_x, max_pupil_x,
                               min_pupil_y, max_pupil_y,
                               min_iris_r, max_iris_r,
                               pixel_step, smooth_sigma, view=False):
    
    pupil_edge = cv2.Canny(pupil_mask, 1, 1)

    pupil = (0, 0, 0)
    circles = cv2.HoughCircles(pupil_edge, cv2.HOUGH_GRADIENT, 1, IRIS_WIDTH, param1=2, param2=1)

    if circles is None:
        return None

    for c in circles[0]:
        if c[2] > pupil[2]:
            pupil = (int(round(c[0])), int(round(c[1])), int(round(c[2])))

    limbus = _fit_circumference(blurred_iris, pupil[2] + 50, pupil[2] * 4,
                                pupil[0] - pupil[2], pupil[0] + pupil[2],  # iris center lies within the pupil (x-axis)
                                pupil[1] - pupil[2], pupil[1] + pupil[2],  # iris center lies within the pupil (y-axis)
                                pixel_step, smooth_sigma)

    if limbus is None:
        return None

    if view:
        iris = cv2.cvtColor(blurred_iris, cv2.COLOR_GRAY2BGR)
        cv2.circle(iris, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 1)
        cv2.circle(iris, (limbus[0], limbus[1]), limbus[2], (0, 255, 0), 1)
        cv2.imshow('Pupil and limbus, press any key.', iris)
        cv2.waitKey(0)

    print('[INFO] Detected limbus and pupillary boundaries.')
    return pupil, limbus


def _04_normalize_iris(iris, pupil, limbus, pupil_mask, specular_reflection_mask, sector_count,
                       norm_iris_width, norm_iris_height, view=False):
    iris_mask = cv2.bitwise_and(pupil_mask, specular_reflection_mask)

    iris_band_width = limbus[2] - pupil[2]

    norm_iris = numpy.zeros((iris_band_width, sector_count), numpy.uint8)
    norm_mask = numpy.zeros((iris_band_width, sector_count), numpy.uint8)

    j = 0
    for angle in numpy.arange(0, 2.0 * numpy.pi, 2.0 * numpy.pi / sector_count):
        sector = numpy.zeros((iris_band_width, 1), numpy.uint8)
        sector_mask = numpy.zeros((iris_band_width, 1), numpy.uint8)

        for i in range(iris_band_width):
            x = int(round(pupil[0] + (pupil[2] + i + 1) * math.sin(angle)))
            y = int(round(pupil[1] + (pupil[2] + i + 1) * math.cos(angle)))

            norm_iris[i, j] = iris[y, x]
            norm_mask[i, j] = iris_mask[y, x]

        j = j + 1

    norm_iris = cv2.resize(norm_iris, (norm_iris_width, norm_iris_height), interpolation=cv2.INTER_CUBIC)
    norm_iris = cv2.equalizeHist(norm_iris)

    norm_mask = cv2.resize(norm_mask, (norm_iris_width, norm_iris_height), interpolation=cv2.INTER_CUBIC)
    _, norm_mask = cv2.threshold(norm_mask, 0, 255, cv2.THRESH_BINARY)

    if view:
        cv2.imshow('Normalized iris, press any key.', norm_iris)
        cv2.waitKey(0)

        cv2.imshow('Normalized iris mask, press any key.', norm_mask)
        cv2.waitKey(0)

    print('[INFO] Obtained normalized version of the given iris.')
    return norm_iris, norm_mask


def enhance(iris, view=False):
    pp_iris = _01_preprocess(iris, output_width=IRIS_WIDTH, view=view)

    blurred_iris, pupil_mask, specr_mask = _02_mask_pupil_and_specular_reflections(pp_iris, IRIS_BLUR_SIZE, view=view)

    pupil_limbus = _03_detect_iris_boundaries(blurred_iris, pupil_mask,
                                              min_pupil_r=MIN_PUPIL_R, max_pupil_r=MAX_PUPIL_R,
                                              min_pupil_x=MIN_PUPIL_X, max_pupil_x=MAX_PUPIL_X,
                                              min_pupil_y=MIN_PUPIL_Y, max_pupil_y=MAX_PUPIL_Y,
                                              min_iris_r=MIN_IRIS_R, max_iris_r=MAX_IRIS_R,
                                              pixel_step=PIXEL_STEP, smooth_sigma=SMOOTH_SIGMA, view=view)

    if pupil_limbus is None:
        return None

    norm_iris, norm_mask = _04_normalize_iris(pp_iris, pupil_limbus[0], pupil_limbus[1], pupil_mask, specr_mask,
                                              NORM_IRIS_SECTOR_COUNT, NORM_IRIS_WIDTH, NORM_IRIS_HEIGHT, view=view)

    return norm_iris, norm_mask
