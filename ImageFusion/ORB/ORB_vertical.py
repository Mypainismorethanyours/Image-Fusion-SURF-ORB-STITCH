import numpy as np
import cv2 as cv

def stitch(img1, img2):
    ORB = cv.ORB_create(500)
    kp1, dst1 = ORB.detectAndCompute(img1, None)
    kp2, dst2 = ORB.detectAndCompute(img2, None)
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(dst1, dst2, None) 
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)
    selected_matches = int(len(matches) * 0.15)
    matches = matches[:selected_matches]
    matching = cv.drawMatches(img1, kp1, img2, kp2, matches, None)
    matching = rotate2(matching)
    cv.imwrite("matching.png", matching)

    p1 = np.zeros((len(matches), 2), dtype=np.float32)
    p2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        p1[i, :] = kp1[match.queryIdx].pt
        p2[i, :] = kp2[match.trainIdx].pt

    (M, mask) = cv.findHomography(p1, p2, cv.RANSAC)
    result = cv.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    return result

def rotate1(img):
    image_in = cv.transpose(img)
    image_out = cv.flip(image_in, 1)
    return image_out

def rotate2(img):
    image_in = cv.transpose(img)
    image_out = cv.flip(image_in, 0)
    return image_out

if __name__ == "__main__":
    img1 = cv.imread('t22.jpg')
    img2 = cv.imread('t21.jpg')
    img1_rotate = rotate1(img1)
    img2_rotate = rotate1(img2)
    output = stitch(img1_rotate, img2_rotate)
    output = rotate2(output)
    cv.imwrite('result.png', output)
