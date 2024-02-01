import sift
import cv2
import numpy as np
def template_match(keypoints1, keypoints2):
    matches = []
    for keypoint1 in keypoints1:
        best_norm = np.inf
        second_best_norm = np.inf
        for keypoint2 in keypoints2:
            des_1 = keypoint1.descriptor
            des_2 = keypoint2.descriptor
            diff_norm = np.linalg.norm(des_1 - des_2)
            if diff_norm < best_norm:
                second_best_norm = best_norm
                best_norm = diff_norm
                possible_match = (keypoint1, keypoint2)
            elif diff_norm < second_best_norm:
                second_best_norm = diff_norm
        if best_norm / second_best_norm < 0.7:
            matches.append(possible_match)
        elif second_best_norm < best_norm:
            raise Exception('second best match is better than best match')
    return matches

import cv2
import numpy as np

def plot_matches(img1, img2, matches, output_path):
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Pad the smaller image
    max_height = max(img1.shape[0], img2.shape[0])
    if img1.shape[0] < max_height:
        padding = np.zeros((max_height - img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
        img1 = np.vstack((img1, padding))
    elif img2.shape[0] < max_height:
        padding = np.zeros((max_height - img2.shape[0], img2.shape[1], 3), dtype=np.uint8)
        img2 = np.vstack((img2, padding))

    img3 = np.hstack((img1, img2))

    # Draw the matches
    for match in matches:
        keypoint1, keypoint2 = match
        # cv2.circle(img3, (keypoint1.j_original, keypoint1.i_original), int(keypoint1.scale), (0, 0, 255), 1)
        # cv2.circle(img3, (keypoint2.j_original + img1.shape[1], keypoint2.i_original), int(keypoint2.scale), (0, 0, 255), 1)
        cv2.line(img3, (keypoint1.j_original, keypoint1.i_original), (keypoint2.j_original + img1.shape[1], keypoint2.i_original), (0, 0, 255), 1)

    cv2.imwrite(output_path, img3)


if __name__ == '__main__':
    dog_img = cv2.imread('/home/paul.yan/csc_420/SIFT/input_image/stat1.jpg')
    dog_keypoints = sift.sift(dog_img, numInterval = 3, numOctive = None, sigma = 1.6, contrastThreshold = 0.03, eigenThreshold = 10)
    dog_face_img = cv2.imread('/home/paul.yan/csc_420/SIFT/input_image/stat2.jpg')
    dog_face_keypoints = sift.sift(dog_face_img, numInterval = 2, numOctive = None, sigma = 1.6, contrastThreshold = 0.03, eigenThreshold = 10)

    sift.plot_keypoints(dog_img, dog_keypoints, '/home/paul.yan/csc_420/SIFT/output_image/dog_keypoints.png')
    sift.plot_keypoints(dog_face_img, dog_face_keypoints, '/home/paul.yan/csc_420/SIFT/output_image/dog_face_keypoints.png')
    matches = template_match(dog_keypoints, dog_face_keypoints)
    plot_matches(dog_img, dog_face_img, matches, '/home/paul.yan/csc_420/SIFT/output_image/dog_matches.png')


    # sift = cv2.SIFT_create()
    # kp = sift.detect(img,None)
    # img=cv2.drawKeypoints(img,kp,img)
    # cv2.imwrite('sift_keypoints.jpg',img)
        