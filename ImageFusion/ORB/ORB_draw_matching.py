# Tool Program: draw ORB_Matching points
import cv2 as cv

def ORB_Matching(img1, img2):
    orb = cv.ORB_create()
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    good_match = []

    bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = bf.match(des1, des2)
    min_distance = matches[0].distance
    max_distance = matches[0].distance

    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_match.append(x)

    draw_match(img1, img2, kp1, kp2, good_match)

def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg = None)
    cv.imwrite("matching.png", outimage)

if __name__ == '__main__':
    img1 = cv.imread('/Users/zhe/Desktop/ORB/imageSet/top/t1.png')
    img2 = cv.imread('/Users/zhe/Desktop/ORB/imageSet/top/t2.png')
    ORB_Matching(img1, img2)
