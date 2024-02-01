import cv2
import numpy as np
import gen_descriptor
import gen_keypoints
                  

def plot_keypoints(img, keypoint_list, output_path):
    img_copy = img.copy()
    for keypoint in keypoint_list:
        cv2.circle(img_copy, (keypoint.j_original, keypoint.i_original), int(keypoint.scale), (0, 0, 255), 1)
    
    cv2.imwrite(output_path, img_copy)

def plotPyrmaid(Pyrmaid):
    for i in range(len(Pyrmaid)):
        for j in range(len(Pyrmaid[i])):
            cv2.imshow(f'blurPyrmaid level {i} scale {j}', Pyrmaid[i][j])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def sift(img, numInterval = 3, numOctive = None, sigma = 1.6, contrastThreshold = 0.03, eigenThreshold = 10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base_image = np.float32(cv2.GaussianBlur(gray, (0, 0), sigmaX = sigma, sigmaY = sigma))
    if numOctive is None:
        numOctive = gen_keypoints.getNumOctive(base_image.shape[0], base_image.shape[1])
    scales = gen_keypoints.getScales(numInterval, sigma)
    blurPyrmaid = gen_keypoints.blurImagePyramid(base_image, numOctive, scales)
    DoGPyrmaid = gen_keypoints.DoG(blurPyrmaid, numInterval)
    extrema_list = gen_keypoints.findLocalExtrema(DoGPyrmaid, contrastThreshold)
    print(f'number of extremas: {len(extrema_list)}')
    key_point_list = gen_keypoints.filterkeypoints(DoGPyrmaid, extrema_list, contrastThreshold, eigenThreshold, sigma, numInterval)
    print(f'number of keypoints: {len(key_point_list)}')

    grad_mag_pyrmaid, grad_ori_pyrmaid = gen_keypoints.GradMagAndOri(blurPyrmaid)
    key_points_with_ori = gen_keypoints.getOrientation(grad_mag_pyrmaid, grad_ori_pyrmaid, key_point_list)
    print(f'number of keypoints with ori: {len(key_points_with_ori)}')

    uni_key_points_with_ori = gen_keypoints.removeDuplicateKeypoints(key_points_with_ori)
    print(f'number of unique keypoints with ori: {len(uni_key_points_with_ori)}')

    gen_descriptor.getDescriptor(grad_mag_pyrmaid, grad_ori_pyrmaid, uni_key_points_with_ori)
    return uni_key_points_with_ori


