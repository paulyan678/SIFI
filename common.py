import numpy as np
from scipy.ndimage.filters import gaussian_filter

def getNeighbourhood(pyrmaid, keypoint):
    octive_index = keypoint.octave_index
    scale_index = keypoint.scale_index
    image = pyrmaid[octive_index][scale_index]
    i, j = keypoint.i_cur_octive, keypoint.j_cur_octive
    img_rows = image.shape[0]
    img_cols = image.shape[1]

    j_start = 0 if j < 7 else j - 7
    j_end = img_cols - 1 if j + 8 >= img_cols else j + 8
    i_start = 0 if i < 7 else i - 7
    i_end = img_rows - 1 if i + 8 >= img_rows else i + 8

    neighbor = image[i_start:i_end + 1, j_start:j_end + 1]


    pad_left = 7 - j if j < 7 else 0
    pad_right = j + 8 - img_cols + 1 if j + 8 >= img_cols else 0
    pad_top = 7 - i if i < 7 else 0
    pad_bottom = i + 8 - img_rows + 1 if i + 8 >= img_rows else 0

    # pad 0 to the neighbor_grad_mag 
    neighbor = np.pad(neighbor, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values = 0.0)

    return neighbor


def getGaussianWeights(scale, mutiplier):
        impulse = np.zeros((16, 16))
        impulse[7, 7] = 1
        gaussian_kernel = gaussian_filter(impulse, 1.5*scale)
        return gaussian_kernel