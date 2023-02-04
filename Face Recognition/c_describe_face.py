import cv2
import numpy
from skimage.feature import local_binary_pattern


X_CELL_DIVISION = 4
Y_CELL_DIVISION = 4

LBP_UNIFORM_PATTERNS = {}
LBP_UNIFORM_PATTERNS[255] = 0
LBP_UNIFORM_PATTERNS[0] = 1

LBP_UNIFORM_PATTERNS[1] = 2
LBP_UNIFORM_PATTERNS[2] = 3
LBP_UNIFORM_PATTERNS[4] = 4
LBP_UNIFORM_PATTERNS[8] = 5
LBP_UNIFORM_PATTERNS[16] = 6
LBP_UNIFORM_PATTERNS[32] = 7
LBP_UNIFORM_PATTERNS[64] = 8
LBP_UNIFORM_PATTERNS[128] = 9

LBP_UNIFORM_PATTERNS[3] = 10
LBP_UNIFORM_PATTERNS[6] = 11
LBP_UNIFORM_PATTERNS[12] = 12
LBP_UNIFORM_PATTERNS[24] = 13
LBP_UNIFORM_PATTERNS[48] = 14
LBP_UNIFORM_PATTERNS[96] = 15
LBP_UNIFORM_PATTERNS[192] = 16
LBP_UNIFORM_PATTERNS[129] = 17

LBP_UNIFORM_PATTERNS[7] = 18
LBP_UNIFORM_PATTERNS[14] = 19
LBP_UNIFORM_PATTERNS[28] = 20
LBP_UNIFORM_PATTERNS[56] = 21
LBP_UNIFORM_PATTERNS[112] = 22
LBP_UNIFORM_PATTERNS[224] = 23
LBP_UNIFORM_PATTERNS[193] = 24
LBP_UNIFORM_PATTERNS[131] = 25

LBP_UNIFORM_PATTERNS[15] = 26
LBP_UNIFORM_PATTERNS[30] = 27
LBP_UNIFORM_PATTERNS[60] = 28
LBP_UNIFORM_PATTERNS[120] = 29
LBP_UNIFORM_PATTERNS[240] = 30
LBP_UNIFORM_PATTERNS[225] = 31
LBP_UNIFORM_PATTERNS[195] = 32
LBP_UNIFORM_PATTERNS[135] = 33

LBP_UNIFORM_PATTERNS[31] = 34
LBP_UNIFORM_PATTERNS[62] = 35
LBP_UNIFORM_PATTERNS[124] = 36
LBP_UNIFORM_PATTERNS[248] = 37
LBP_UNIFORM_PATTERNS[241] = 38
LBP_UNIFORM_PATTERNS[227] = 39
LBP_UNIFORM_PATTERNS[199] = 40
LBP_UNIFORM_PATTERNS[143] = 41

LBP_UNIFORM_PATTERNS[63] = 42
LBP_UNIFORM_PATTERNS[126] = 43
LBP_UNIFORM_PATTERNS[252] = 44
LBP_UNIFORM_PATTERNS[249] = 45
LBP_UNIFORM_PATTERNS[243] = 46
LBP_UNIFORM_PATTERNS[231] = 47
LBP_UNIFORM_PATTERNS[207] = 48
LBP_UNIFORM_PATTERNS[159] = 49

LBP_UNIFORM_PATTERNS[127] = 50
LBP_UNIFORM_PATTERNS[254] = 51
LBP_UNIFORM_PATTERNS[253] = 52
LBP_UNIFORM_PATTERNS[251] = 53
LBP_UNIFORM_PATTERNS[247] = 54
LBP_UNIFORM_PATTERNS[239] = 55
LBP_UNIFORM_PATTERNS[223] = 56
LBP_UNIFORM_PATTERNS[191] = 57

LBP_NON_UNIFORM = 58


def _01_divide_into_cells(enhanced_face, x_cell_division, y_cell_division, view=False):
    h, w = enhanced_face.shape

    x_offset = int(round(w / x_cell_division))
    y_offset = int(round(h / y_cell_division))

    cells = []
    for row in range(0, h, x_offset):
        for col in range(0, w, y_offset):
            cells.append(enhanced_face[row:row + x_offset, col:col + y_offset])

    if view and len(cells) > 0:
        img = numpy.concatenate(cells, axis=0)
        cv2.imshow('Obtained cells, press key.', img)
        cv2.waitKey(0)

    print('[INFO] Divided the face into', len(cells), 'cells.')
    return cells


def _02_lbp_describe_cells(cells, view=False):
    lbps = []

    for cell in cells:
        lbps.append(local_binary_pattern(cell, 8, 1))  # "8" neighbors, "1" pixel away from center (3x3 neighborhood)

    if view and len(lbps) > 0:
        img = numpy.concatenate(lbps, axis=0)
        img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow('Local binary patterns, press key.', img)
        cv2.waitKey(0)

    print('[INFO] Computed the', len(lbps), 'local binary patterns (LBP).')
    return lbps


def _03_map_lbp_to_uniform_pattern(lbps, view=False):
    maps = []

    for lbp in lbps:
        map = numpy.zeros(lbp.shape) + LBP_NON_UNIFORM

        h, w = lbp.shape
        for row in range(h):
            for col in range(w):
                key = lbp[row, col]
                if key in LBP_UNIFORM_PATTERNS.keys():
                    map[row, col] = LBP_UNIFORM_PATTERNS[key]

        maps.append(map)

    if view and len(maps) > 0:
        img = numpy.concatenate(maps, axis=0)
        img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow('Uniform patterns, press key.', img)
        cv2.waitKey(0)

    print('[INFO] Computed the', len(maps), 'uniform LBP patterns.')
    return maps


def _04_compute_histograms(uniform_lbp_maps):
    histograms = []

    for map in uniform_lbp_maps:
        hist, l = numpy.histogram(map, bins=numpy.arange(0, 60))

        hist_sum = numpy.sum(hist)
        hist = hist / hist_sum

        histograms.append(hist)

    print('[INFO] Computed the', len(histograms), 'cell-wise histograms of the uniform LBP patterns.')
    return histograms


def describe(enhanced_face, view=False):
    face_cells = _01_divide_into_cells(enhanced_face, X_CELL_DIVISION, Y_CELL_DIVISION, view=True)

    lbps = _02_lbp_describe_cells(face_cells, view=view)

    uniform_lbp_maps = _03_map_lbp_to_uniform_pattern(lbps, view=view)

    histograms = _04_compute_histograms(uniform_lbp_maps)

    return numpy.concatenate(histograms, axis=0)
