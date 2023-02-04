import numpy

def match(face_description_1, face_description_2):
    dissimilarity = numpy.linalg.norm(face_description_1 - face_description_2, ord=2)  # '2': l2, Euclidean distance
    similarity = 1 - dissimilarity
    return similarity
