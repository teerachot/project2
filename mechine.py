import numpy as np
from sklearn import svm
from sklearn import datasets
from skimage.feature import hog
import cv2

def test_datafrom(file):
	f = open(file,"r")
	label = []
	test = []
	for line in f.readline():
		img = []
		for pix in line:
			if pix.isalnum():
				label.append(pix)
			if pix.isalnum():
				img.append(pix)
				

def main():
	test, test_label = test_datafrom("mnist_train_for_read_from_py.csv")
	dataset = datasets.fetch_mldata("MNIST Original")
	data = np.array(dataset.data,"int16")
	label = np.array(dataset.target,"int8")
	list_hog_img = []
	for img in data:
		img1 = hog(img.reshape((28, 28)), orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualise=False)
		list_hog_img.append(img1)
	data_imgs = np.array(list_hog_img, 'float64')
	
	machine = LinearSVC()
	machine.trian(data_imgs,label)
	machine.save("")

if __name__ == '__main__':
	main()