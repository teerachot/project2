import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog


def main():
    # img = cv2.imread("13.jpg")
    img = cv2.imread("555.png")
    img = cv2.resize(img, (768, 546))
    knn = joblib.load("digits_knn.pkl")
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_img = cv2.GaussianBlur(g_img, (9, 9,), 0)
    # cv2.imshow("a",g_img)
    # cv2.waitKey(0)
    ret, thres = cv2.threshold(g_img, 130, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("s", thres)
    cv2.waitKey(0)
    im, contours, hira = cv2.findContours(
        thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in contours]
    iop = 0
    for x, y, w, h in rects:
        if h < 100 and w < 100:
            continue
            # print('x: {:^2}'.format(x))
            # print('y: {:^2}'.format(y))
            # print('w: {:^2}'.format(w))
            # print('h: {:^2}'.format(h))
            # print("-------------------------")
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 255), 3)
        # leng = int(h * 1)
        # plt1 = abs(int(y + h // 2 - leng // 2))
        # plt2 = abs(int(x + w // 2 - leng // 2))
        # print(leng)
        # print(plt1)
        # print(plt2)
        # roi = thres[plt1:plt1 + leng, plt2:plt2 + leng]
        roi = thres[abs(y - 20):y + h + 20, abs(x - 20):x + w + 20]

        # cv2.imshow('test', roi)
        # cv2.waitKey(0)

        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        cv2.imshow('test', roi)
        cv2.waitKey(0)
        roi = cv2.dilate(roi, (3, 3))
        cv2.imwrite("roi" + '{}.png'.format(iop), roi)
        iop = iop + 1

        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(
            14, 14), cells_per_block=(1, 1), visualise=False)
        num = knn.predict(np.array([roi_hog_fd], 'float64'))
        print(num)
        cv2.putText(img, str(num[0]), (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 3)

    cv2.imshow('555', img)
    cv2.waitKey(0)
    cv2.imwrite('result.jpg', img)

if __name__ == '__main__':
    main()
