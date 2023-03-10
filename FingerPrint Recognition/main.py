import a_acquire
import b_enhance
import c_describe
import d_match

fingeprint_filepath_1 = './test-data/1.bmp'
fingeprint_filepath_2 = './test-data/2.png'

fingerprint_1 = a_acquire.acquire_from_file(fingeprint_filepath_1, view=False)
fingerprint_2 = a_acquire.acquire_from_file(fingeprint_filepath_2, view=False)

pp_fingerprint_1, en_fingerprint_1, mask_1 = b_enhance.enhance(fingerprint_1, dark_ridges=False, view=False)
pp_fingerprint_2, en_fingerprint_2, mask_2 = b_enhance.enhance(fingerprint_2, dark_ridges=False, view=False)

ridge_endings_1, bifurcations_1 = c_describe.describe(en_fingerprint_1, mask_1, view=False)
ridge_endings_2, bifurcations_2 = c_describe.describe(en_fingerprint_2, mask_2, view=False)

match = d_match.match(en_fingerprint_1, ridge_endings_1, bifurcations_1, en_fingerprint_2, ridge_endings_2, bifurcations_2, view=True)
