import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.metrics import accuracy_score
import time


def test_datafrom(file, label, n):
    f = open(file, "rb")
    l = open(label, "rb")
    f.read(16)
    l.read(8)
    label = []
    test = []
    for i in range(n):
        label.append(ord(l.read(1)))
        img = []
        for j in range(28 * 28):
            img.append(ord(f.read(1)))
        test.append(img)
        # if i == 50000:
        #     break
    return np.array(test, "int16"), np.array(label, "int8")


def main():
    knn = joblib.load("digits_knn.pkl")
    Linear = joblib.load("digits_linear.pkl")
    LinearSVC = joblib.load("digits_LinearSVC.pkl")
    poly = joblib.load("digits_poly.pkl")
    rbf = joblib.load("digits_rbf.pkl")
    test, label = test_datafrom(
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 9900)
    test1 = []
    for img in test:
        img = hog(img.reshape((28, 28)), orientations=9, pixels_per_cell=(
            14, 14), cells_per_block=(1, 1), visualise=False)
        test1.append(img)

    test = np.array(test1, 'float64')

    start_time = time.time()
    result_knn = knn.predict(test)
    time_knn = time.time() - start_time

    start_time = time.time()
    result_Linear = Linear.predict(test)
    time_Linear = time.time() - start_time

    start_time = time.time()
    result_LinearSVC = LinearSVC.predict(test)
    time_LinearSVC = time.time() - start_time

    start_time = time.time()
    result_poly = poly.predict(test)
    time_poly = time.time() - start_time

    start_time = time.time()
    result_rbf = rbf.predict(test)
    time_rbf = time.time() - start_time

    acc_knn = accuracy_score(label, result_knn) * 100
    acc_Linear = accuracy_score(label, result_Linear) * 100
    acc_LinearSVC = accuracy_score(label, result_LinearSVC) * 100
    acc_poly = accuracy_score(label, result_poly) * 100
    acc_rbf = accuracy_score(label, result_rbf) * 100

    report = open("report.txt", "w")
    report.write("=======amount data 9900 data digits")
    report.write("---time predict----\n")
    print("----time predict-----")
    print("------SVM------")
    print("time_Linear: %f sec" % (time_Linear))
    print("time_rbf: %f sec" % (time_rbf))
    print("time_poly: %f sec" % (time_poly))
    print("time_LinearSVC: %f sec" % (time_LinearSVC))
    print("---------------")
    print("time_knn: %f sec" % (time_knn))
    print()
    report.write("------SVM------\n")
    report.write("time_Linear: " + str(float(time_Linear)) + "sec \n")
    report.write("time_rbf: " + str(float(time_rbf)) + "sec \n")
    report.write("time_poly: " + str(float(time_poly)) + "sec \n")
    report.write("time_LinearSVC: " + str(float(time_LinearSVC)) + "sec \n")
    report.write("---------------\n")
    report.write("time_knn: " + str(float(time_knn)) + "sec \n")
    report.write("\n")
    
    report.write("---Accuracy_score----\n")
    print("---Accuracy_score----")
    report.write("---Accuracy_score----\n")
    print("------SVM------")
    print("linear: %f " % (acc_Linear))
    print("rbf: %f " % (acc_rbf))
    print("poly: %f " % (acc_poly))
    print("LinearSVC: %f " % (acc_LinearSVC))
    print("---------------")
    print("K-NN: %f " % (acc_knn))
    report.write("------SVM------\n")
    report.write("linear: " + str(acc_Linear) + "\n")
    report.write("rbf: " + str(acc_rbf) + "\n")
    report.write("poly: " + str(acc_poly) + "\n")
    report.write("LinearSVC: " + str(acc_LinearSVC) + "\n")
    report.write("---------------\n")
    report.write("K-NN: " + str(acc_knn) + "\n")
    report.close()

if __name__ == '__main__':
    main()
