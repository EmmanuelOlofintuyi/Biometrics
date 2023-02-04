# Iris Recognition
# 04. Module to match iris descriptions.
# Language: Python 3

import numpy
import cv2


ROTATIONS = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def _rotate_norm_image(image, rotation):
    output = numpy.zeros(image.shape, image.dtype)

    if rotation == 0:
        return image

    else:
        output[:, rotation:] = image[:, :-rotation]
        output[:, :rotation] = image[:, -rotation:]

    return output


def _compute_norm_hamming_distance(description_1, mask_1, description2, mask_2):
    comb_mask = cv2.bitwise_and(mask_1, mask_2)

    bit_up_count = numpy.sum(comb_mask > 0)

    xor_output = cv2.bitwise_xor(description_1, description2)
    xor_output = cv2.bitwise_and(xor_output, xor_output, mask=comb_mask)
    dist = numpy.sum(xor_output > 0)

    return float(dist) / bit_up_count


def match(descriptions_1, mask_1, descriptions_2, mask_2):
    rot_distances = []

    for rotation in ROTATIONS:
        distances = []

        for i in range(len(descriptions_1)):  # could be "for i in range(len(descriptions_2)):"
            desc_1 = descriptions_1[i]
            rot_desc_2 = _rotate_norm_image(descriptions_2[i], rotation)
            rot_mask_2 = _rotate_norm_image(mask_2, rotation)

            distances.append(_compute_norm_hamming_distance(desc_1, mask_1, rot_desc_2, rot_mask_2))

        rot_distances.append(numpy.mean(distances))

    print('[INFO] Computed normalized Hamming distance.')
    return numpy.min(rot_distances)
