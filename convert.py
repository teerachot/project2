
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(18)
    print()
    l.read(8)
    images = []
    count = 0
    count1 = 0
    for i in range(n):
        label = ord(l.read(1))
        image = [label]
        print(label)
        count1 = count1 + 1
        for j in range(28 * 28):
            count = count + 1
            num = ord(f.read(1))
            print('{:^3}'.format(num), end=" ")
            if count % 28 == 0:
                print()
            image.append(num)
        images.append(image)
        if count1 == 9000:
            break
    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")

    f.close()
    o.close()
    l.close()

# convert("train-images.idx3-ubyte", "train-labels-idx1-ubyte",
#         "mnist_train_for_read_from_py.csv", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
        "mnist_test.csv", 10000)
