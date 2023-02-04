# Iris Recognition
# 03. Module to describe iris texture.
# Language: Python 3

import math
import numpy
import cv2
import scipy.ndimage


BIN_THRESH = 0.5


def _01_load_bsif_filters(filter_file_path, view=False):
    values = []
    with open(filter_file_path) as f:
        for line in f:
            parts = line.strip().split(',')
            for i in range(len(parts)):
                if len(values) == i:
                    values.append([])

                values[i].append(float(parts[i]))

    filters = []
    filter_size = int(math.sqrt(len(values[0])))

    for vector in values:
        filters.append(numpy.reshape(vector, (filter_size, filter_size)))

        if view:
            view_filter = numpy.zeros((filter_size, filter_size))
            view_filter = cv2.normalize(filters[-1], view_filter, 0, 255, cv2.NORM_MINMAX).astype(numpy.uint8)
            cv2.imshow('Current filter, press any key.', view_filter)
            cv2.waitKey(0)

    print('[INFO] Loaded BSIF filters.')
    return filters


def _02_bsif_describe(iris, filters, threshold, view=False):
    output = []

    for f in filters:
        conv = scipy.ndimage.convolve(iris, f)
        conv = cv2.normalize(conv, conv, -1.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32FC1)
        _, description = cv2.threshold(conv, threshold, 255, cv2.THRESH_BINARY)
        output.append(description.astype(numpy.uint8))

        if view:
            view_conv = numpy.zeros(conv.shape)
            view_conv = cv2.normalize(description, view_conv, 0, 255, cv2.NORM_MINMAX).astype(numpy.uint8)
            cv2.imshow('Filtered iris, press any key.', view_conv)
            cv2.waitKey(0)

    print('[INFO] Described given iris.')
    return output


def describe(enhanced_iris, filter_file_path, view=False):
    filters = _01_load_bsif_filters(filter_file_path, view=view)

    descriptions = _02_bsif_describe(enhanced_iris, filters, BIN_THRESH, view=view)

    return descriptions
