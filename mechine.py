import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib 
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
import time
import cv2


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
    return np.array(test,"int16"), np.array(label,"int8")


def main():
    test, test_label = test_datafrom(
        "train-images.idx3-ubyte", "train-labels-idx1-ubyte", 60000)
    # print(test)
    # print(test_label)
    dataset = datasets.fetch_mldata("MNIST Original")
    data = np.array(dataset.data,"int16")
    label = np.array(dataset.target,"int8")
    list_hog_img = []
    test1= []
    for img in data:
    	img1 = hog(img.reshape((28, 28)), orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualise=False)
    	list_hog_img.append(img1)
    for img in test:
    	img = hog(img.reshape((28, 28)), orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualise=False)
    	test1.append(img)
    test = np.array(test1,'float64')
    data_imgs = np.array(list_hog_img, 'float64')
    machine1 = svm.SVC(kernel='linear',C=1)
    machine2 = svm.SVC(kernel='rbf', gamma=0.7, C=1)
    machine3 = svm.SVC(kernel='poly',degree= 3, C=1 )
    machine4 = svm.LinearSVC(C=1)
    machine5 = KNeighborsClassifier()
    write_flie = open("result","w")
    write_flie.write("------------result------------\n")
    # -----------------1--------------------
    start_time = time.time()
    machine1.fit(data_imgs,label)
    timer = time.time()-start_time
    write_flie.write("svm linear  "+str(timer)+"seconds"+"\n")
    print("-linear----%s seconds----"%(timer))
    # -----------------2--------------------
    start_time = time.time()
    machine2.fit(data_imgs,label)
    timer = time.time()-start_time
    write_flie.write("svm rbf  "+str(timer)+"seconds"+"\n")
    print("--rbf---%s seconds----"%(time.time()-start_time))
    # -----------------3---------------------
    start_time = time.time()
    machine3.fit(data_imgs,label)
    timer = time.time()-start_time
    write_flie.write("svm poly  "+str(timer)+"seconds"+"\n")
    print("-poly----%s seconds----"%(time.time()-start_time))
    # -----------------4--------------------
    start_time = time.time()
    machine4.fit(data_imgs,label)
    timer = time.time()-start_time
    write_flie.write("svm LibearSVC  "+str(timer)+"seconds"+"\n")
    print("--LinearSVC---%s seconds----"%(time.time()-start_time))
    # -----------------5--------------------
    start_time = time.time()
    machine5.fit(data_imgs,label)
    timer = time.time()-start_time
    write_flie.write("K-nn  "+str(timer)+"seconds"+"\n")
    print("--K-NN---%s seconds----"%(time.time()-start_time))
    
    joblib.dump(machine1,"digits_linear.pkl",compress=3)
    joblib.dump(machine2,"digits_rbf.pkl",compress=3)
    joblib.dump(machine3,"digits_poly.pkl",compress=3)
    joblib.dump(machine4,"digits_LinearSVC.pkl",compress=3)
    joblib.dump(machine5,"digits_knn.pkl",compress=3)

    write_flie.close()
    # resuit = machine.predict(test[:10000])
    # count = 0
    # print(resuit)
    # print(test_label[:10000])
    # for x in range(10000):
    # 	if test_label[x]==resuit[x]:
    # 		count =count +1
    # print(count)
    # # machine.save("")

if __name__ == '__main__':
    main()