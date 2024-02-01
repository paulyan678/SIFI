import numpy as np
from common import getGaussianWeights, getNeighbourhood

def inverse_trilinear_interpolation(x, y, z, value, dest):

    x1, y1, z1 = np.floor(x).astype(int), np.floor(y).astype(int), np.floor(z).astype(int)
    x2, y2, z2 = np.ceil(x).astype(int), np.ceil(y).astype(int), np.ceil(z).astype(int)

    xd = (x - x1) / (x2 - x1)
    yd = (y - y1) / (y2 - y1)
    zd = (z - z1) / (z2 - z1)

    dest[x1, y1, z1] += value*(1-xd)*(1-yd)*(1-zd)

    if z2 < 8:
        dest[x1, y1, z2] += value*(1-xd)*(1-yd)*zd

    if y2 < 8:
        dest[x1, y2, z1] += value*(1-xd)*yd*(1-zd)

    if z2 < 8 and y2 < 8:
        dest[x1, y2, z2] += value*(1-xd)*yd*zd

    if x2 < 8:
        dest[x2, y1, z1] += value*xd*(1-yd)*(1-zd)

    if x2 < 8 and z2 < 8:
        dest[x2, y1, z2] += value*xd*(1-yd)*zd

    if x2 < 8 and y2 < 8:
        dest[x2, y2, z1] += value*xd*yd*(1-zd)
    
    if x2 < 8 and y2 < 8 and z2 < 8:
        dest[x2, y2, z2] += value*xd*yd*zd

def getDescriptor(grad_mag_pyrmaid, grad_ori_pyrmaid, key_point_list):

    for keypoint in key_point_list:
        descriptor = np.zeros((4, 4, 8))  # There are 4x4 sub-blocks, each with 8 bins

        # Assuming getNeighbourhood and getGaussianWeights functions are defined elsewhere
        neighbor_grad_mag = getNeighbourhood(grad_mag_pyrmaid, keypoint)
        neighbor_grad_ori = getNeighbourhood(grad_ori_pyrmaid, keypoint)

        gaussian_kernel = getGaussianWeights(keypoint.scale, 0.5)
        weights = neighbor_grad_mag * gaussian_kernel

        dominant_ori = keypoint.orientation

        for i in range(-7, 9):
            for j in range(-7, 9):
                # Rotate the coordinates
                rot_i = i * np.cos(dominant_ori) + j * np.sin(dominant_ori)
                rot_j = i * -1 * np.sin(dominant_ori) + j * np.cos(dominant_ori)

                # Translate coordinates to have (0,0) at the top left corner of the 16x16 patch
                trans_i = rot_i + 7
                trans_j = rot_j + 7

                # Determine the sub-block (4x4) in which the coordinate falls
                block_i = trans_i / 4
                block_j = trans_j / 4

                # Ensure the block indices are within the 4x4 grid
                if 0.0 <= block_i <= 3.0 and 0.0 <= block_j <= 3.0:
                    # Calculate the bin index based on the orientation
                    # Assuming orientations are in radians and within [0, 2*pi)
                    diff = neighbor_grad_ori[i + 7, j + 7] - dominant_ori
                    if diff < 0.0:
                        diff += 360.0

                    bin_index = diff / 45.0

                    inverse_trilinear_interpolation(block_i, block_j, bin_index, weights[i + 7, j + 7], descriptor)

        flatten_descriptor = descriptor.flatten()
        flatten_descriptor = flatten_descriptor / np.sqrt(np.sum(flatten_descriptor ** 2) + 1e-6)
        flatten_descriptor[flatten_descriptor > 0.2] = 0.2
        flatten_descriptor = flatten_descriptor / np.sqrt(np.sum(flatten_descriptor ** 2) + 1e-6)
        keypoint.descriptor = flatten_descriptor

