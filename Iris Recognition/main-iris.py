import a_acquire_iris
import b_enhance_iris
import c_describe_iris
import d_match_iris

iris_filepath_1 = './test-images-iris/iris1.bmp'
iris_filepath_2 = './test-images-iris/iris2.bmp'

filer_filepath = './filters-iris/filters.txt'

iris_1 = a_acquire_iris.acquire_from_file(iris_filepath_1, view=False)
iris_2 = a_acquire_iris.acquire_from_file(iris_filepath_2, view=False)

norm_iris_1, norm_mask_1 = b_enhance_iris.enhance(iris_1, view=False)
norm_iris_2, norm_mask_2 = b_enhance_iris.enhance(iris_2, view=False)

descriptions_1 = c_describe_iris.describe(norm_iris_1, filer_filepath, view=False)
descriptions_2 = c_describe_iris.describe(norm_iris_2, filer_filepath, view=False)

distance_12 = d_match_iris.match(descriptions_1, norm_mask_1, descriptions_2, norm_mask_2)
print('Distance 12:', distance_12)

distance_21 = d_match_iris.match(descriptions_2, norm_mask_2, descriptions_1, norm_mask_1)
print('Distance 21:', distance_21)
