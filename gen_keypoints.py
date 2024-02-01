
import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import correlate2d, correlate
from functools import cmp_to_key
from common import getNeighbourhood, getGaussianWeights


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

class myKeypoint:
    def __init__(self, i_original, j_original, i_cur_octive, j_cur_octive, octave_index, scale_index, scale, responds, orientation, descriptor):
        self.i_original = i_original
        self.j_original = j_original
        self.i_cur_octive = i_cur_octive
        self.j_cur_octive = j_cur_octive
        self.octave_index = octave_index
        self.scale_index = scale_index
        self.scale = scale
        self.responds = responds
        self.orientation = orientation
        self.descriptor = descriptor   

def getNumOctive(height, wodth):
    return int(np.floor(np.log2(min(height, wodth))))

def getScales(numInterval, sigma):
    k = 2 ** (1/numInterval)
    numImagesPerOctive = numInterval + 3
    scales = np.zeros(numImagesPerOctive)
    scales[0] = sigma
    for i in range(1, numImagesPerOctive):
        sigmaPre = (k ** (i - 1)) * sigma
        sigmaCur = sigmaPre * k
        scales[i] = np.sqrt(sigmaCur ** 2 - sigmaPre ** 2)
    return scales

def blurImagePyramid(img, numOctive, scales):
    blurPyrmaid = []
    baseOctive = img
    for i in range(numOctive):
        blurOctive = np.zeros((len(scales), baseOctive.shape[0], baseOctive.shape[1]))
        blurOctive[0] = baseOctive
        for j in range(1, len(scales)):
            blurOctive[j] = cv2.GaussianBlur(blurOctive[j - 1], (0, 0), sigmaX = scales[j], sigmaY = scales[j])
        blurPyrmaid.append(blurOctive)
        baseOctive = cv2.resize(blurOctive[-3], None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_NEAREST)

    return blurPyrmaid

def DoG(blurPyrmaid, numInterval):
    DoGPyrmaid = []
    num_dog_per_octive = numInterval + 2
    for i in range(len(blurPyrmaid)):
        octave = blurPyrmaid[i]
        DoGOctive = octave[1:] - octave[:-1]
        DoGPyrmaid.append(DoGOctive)
    return DoGPyrmaid

def findLocalExtrema(DoGPyrmaid, contrastThreshold):

    extrema_list = []
    for octive_index, octive_dog in enumerate(DoGPyrmaid):
        abs_octive_dog = np.abs(octive_dog)
        maxium_vals = maximum_filter(abs_octive_dog, (3, 3, 3), mode='wrap')
        max_mask = abs_octive_dog == maxium_vals
        #filter out the ma values that is smaller than threashold
        max_mask = max_mask & (abs_octive_dog > 255 * contrastThreshold)
        # convert to a numppy array of coordinates
        max_list = np.argwhere(max_mask)

        for scale_index, i, j in max_list:
            extrema_list.append(myKeypoint(None, None, i, j, octive_index, scale_index, None, None, None, None))

    return extrema_list                        

def GradMagAndOri(blurPyrmaid):
    grad_mag_pyrmaid = []
    grad_ori_pyrmaid = []
    for octive in blurPyrmaid:
        numImg, rows, cols = octive.shape
        reshaped_octive = octive.reshape((rows, numImg * cols))
        grad_x_reshaped = correlate2d(reshaped_octive, SOBEL_X, mode = 'same')
        grad_y_reshaped = correlate2d(reshaped_octive, SOBEL_Y, mode = 'same')
        grad_mag_reshaped = np.sqrt(grad_x_reshaped ** 2 + grad_y_reshaped ** 2)

        degrees = np.degrees(np.arctan2(grad_y_reshaped, grad_x_reshaped))
        degrees[degrees < 0.0] += 360
        grad_ori_reshaped = degrees

        grad_mag = grad_mag_reshaped.reshape((numImg, rows, cols))
        grad_ori = grad_ori_reshaped.reshape((numImg, rows, cols))
        grad_mag_pyrmaid.append(grad_mag)
        grad_ori_pyrmaid.append(grad_ori)
    return grad_mag_pyrmaid, grad_ori_pyrmaid



def getOrientation(grad_mag_pyrmaid, grad_ori_pyrmaid, key_point_list):
    key_point_list_with_ori = []
    for keypoint in key_point_list:
        raw_histogram = np.zeros(36)
        neighbor_grad_mag = getNeighbourhood(grad_mag_pyrmaid, keypoint)
        neighbor_grad_ori = getNeighbourhood(grad_ori_pyrmaid, keypoint)

        # get the weighting gaussian kernel
        gaussian_kernel = getGaussianWeights(keypoint.scale, 1.5)

        weights = neighbor_grad_mag * gaussian_kernel

        # update the histogram
        for i in range(16):
            for j in range(16):
                bin_index = int(np.floor(neighbor_grad_ori[i, j] / 10))
                raw_histogram[bin_index] += weights[i, j]
    
        smoothed_histogram = correlate(raw_histogram, np.array([1, 4, 6, 4, 1])/16, mode = 'same')

        max_value = np.max(smoothed_histogram)
        local_max_values = maximum_filter(smoothed_histogram, 3, mode = 'wrap')
        local_max_mask = smoothed_histogram == local_max_values
        local_max_mask = local_max_mask & (smoothed_histogram > 0.8 * max_value)
        local_max_list = np.argwhere(local_max_mask).flatten()

        for local_max_index in local_max_list:
            left_value = smoothed_histogram[(local_max_index - 1) % 36]
            right_value = smoothed_histogram[(local_max_index + 1) % 36]
            interpolated_index = (local_max_index + 0.5 * (left_value - right_value) / (left_value - 2 * smoothed_histogram[local_max_index] + right_value)) % 36
            orientation = np.float32(interpolated_index * 10)
            if orientation > 360.0:
                raise Exception('orientation is greater than 360')
            key_point_list_with_ori.append(myKeypoint(keypoint.i_original, 
                                                      keypoint.j_original, 
                                                      keypoint.i_cur_octive, 
                                                      keypoint.j_cur_octive, 
                                                      keypoint.octave_index, 
                                                      keypoint.scale_index, 
                                                      keypoint.scale, 
                                                      keypoint.responds, 
                                                      orientation, 
                                                      None))
    return key_point_list_with_ori

# modify the code in doc string so it can be used in this project for the class type myKeypoint
def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.i_original != keypoint2.i_original:
        return keypoint1.i_original - keypoint2.i_original
    if keypoint1.j_original != keypoint2.j_original:
        return keypoint1.j_original - keypoint2.j_original
    if keypoint1.j_cur_octive != keypoint2.j_cur_octive:
        return keypoint1.j_cur_octive - keypoint2.j_cur_octive
    if keypoint1.i_cur_octive != keypoint2.i_cur_octive:
        return keypoint1.i_cur_octive - keypoint2.i_cur_octive
    if keypoint1.scale != keypoint2.scale:
        return keypoint2.scale - keypoint1.scale
    if keypoint1.orientation != keypoint2.orientation:
        return keypoint1.orientation - keypoint2.orientation
    if keypoint1.responds != keypoint2.responds:
        return keypoint2.responds - keypoint1.responds
    if keypoint1.octave_index != keypoint2.octave_index:
        return keypoint2.octave_index - keypoint1.octave_index
    return keypoint2.scale_index - keypoint1.scale_index

def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.i_original != next_keypoint.i_original or \
           last_unique_keypoint.j_original != next_keypoint.j_original or \
           last_unique_keypoint.j_cur_octive != next_keypoint.j_cur_octive or \
           last_unique_keypoint.i_cur_octive != next_keypoint.i_cur_octive or \
           last_unique_keypoint.scale != next_keypoint.scale or \
           last_unique_keypoint.orientation != next_keypoint.orientation or \
           last_unique_keypoint.responds != next_keypoint.responds or \
           last_unique_keypoint.octave_index != next_keypoint.octave_index or \
           last_unique_keypoint.scale_index != next_keypoint.scale_index:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

def filterkeypoints(DoGPyrmaid, extrema_list, contrastThreshold, eigenThreshold, sigma, numInterval):
    key_point_list = []
    for keypoint in extrema_list:
        i = keypoint.i_cur_octive
        j = keypoint.j_cur_octive
        octave_index = keypoint.octave_index
        scale_index = keypoint.scale_index
        gradient_vec = getGradient(DoGPyrmaid, i, j, octave_index, scale_index)
        hessian_mat = getHessian(DoGPyrmaid, i, j, octave_index, scale_index)

        num_scales, num_rows, num_cols = DoGPyrmaid[octave_index].shape
        i_offset, j_offset, scale_offset = -np.linalg.lstsq(hessian_mat, gradient_vec, rcond = None)[0]
        updated_i = (i + int(round(i_offset))) % num_rows
        updated_j = (j + int(round(j_offset))) % num_cols
        updated_scale_index = (scale_index + int(round(scale_offset))) % num_scales

        responds_updated = DoGPyrmaid[octave_index][scale_index, i, j] / 255 + 0.5 * np.dot(gradient_vec, np.array([i_offset, j_offset, scale_offset]))
        if responds_updated < contrastThreshold:
            continue

        edge_hessian = hessian_mat[:2, :2]
        trace = np.trace(edge_hessian)
        det = np.linalg.det(edge_hessian)
        if det <= 0 or (trace ** 2 / det) >= ((eigenThreshold + 1) ** 2 / eigenThreshold):
            continue

        keypoint.i_cur_octive = updated_i
        keypoint.j_cur_octive = updated_j
        keypoint.i_original = updated_i * (2 ** octave_index)
        keypoint.j_original = updated_j * (2 ** octave_index)
        keypoint.scale_index = updated_scale_index
        keypoint.scale = sigma * (2 ** (octave_index + updated_scale_index / numInterval))
        keypoint.responds = responds_updated
        key_point_list.append(keypoint)

    return key_point_list

def getGradient(DoGPyrmaid, i, j, octave_index, scale_index):
    octive_dog = DoGPyrmaid[octave_index]
    num_scale, num_rows, num_cols = octive_dog.shape

    # rewrite like the getHessian function
    i_next = (i + 1) % num_rows
    i_pre = (i - 1) % num_rows
    j_next = (j + 1) % num_cols
    j_pre = (j - 1) % num_cols
    scale_index_pre = (scale_index - 1) % num_scale
    scale_index_next = (scale_index + 1) % num_scale

    di = (octive_dog[scale_index][i_next, j] - octive_dog[scale_index][i_pre, j]) / (2*255)
    dj = (octive_dog[scale_index][i, j_next] - octive_dog[scale_index][i, j_pre]) / (2*255)
    ds = (octive_dog[scale_index_next][i, j] - octive_dog[scale_index_pre][i, j]) / (2*255)
    return np.array([di, dj, ds])

def getHessian(DoGPyrmaid, i, j, octave_index, scale_index):
    octive_dog = DoGPyrmaid[octave_index]
    num_scale, num_rows, num_cols = octive_dog.shape
    i_next = (i + 1) % num_rows
    i_pre = (i - 1) % num_rows
    j_next = (j + 1) % num_cols
    j_pre = (j - 1) % num_cols
    pre_scale_index = (scale_index - 1) % num_scale
    next_scale_index = (scale_index + 1) % num_scale

    dii = (octive_dog[scale_index][i_next, j] + octive_dog[scale_index][i_pre, j] - 2 * octive_dog[scale_index][i, j]) / 255
    djj = (octive_dog[scale_index][i, j_next] + octive_dog[scale_index][i, j_pre] - 2 * octive_dog[scale_index][i, j]) / 255
    dss = (octive_dog[next_scale_index][i, j] + octive_dog[pre_scale_index][i, j] - 2 * octive_dog[scale_index][i, j]) /255
    dij = (octive_dog[scale_index][i_next, j_next] - octive_dog[scale_index][i_next, j_pre] - octive_dog[scale_index][i_pre, j_next] + octive_dog[scale_index][i_pre, j_pre]) / (4*255)
    dis = (octive_dog[next_scale_index][i_next, j] - octive_dog[next_scale_index][i_pre, j] - octive_dog[pre_scale_index][i_next, j] + octive_dog[pre_scale_index][i_pre, j]) / (4*255)
    djs = (octive_dog[next_scale_index][i, j_next] - octive_dog[next_scale_index][i, j_pre] - octive_dog[pre_scale_index][i, j_next] + octive_dog[pre_scale_index][i, j_pre]) / (4*255)



    return np.array([[dii, dij, dis], [dij, djj, djs], [dis, djs, dss]])




