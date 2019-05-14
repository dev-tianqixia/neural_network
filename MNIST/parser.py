import numpy as np
from matplotlib import pyplot as plt

class Parser:
    def parse(self, image_file_path, label_file_path):
        images = []
        with open(image_file_path, 'rb') as f:
            # skip the magic number
            f.seek(4)
            # read meta data
            self.image_count = int.from_bytes(f.read(4), byteorder='big')
            self.rows = int.from_bytes(f.read(4), byteorder='big')
            self.cols = int.from_bytes(f.read(4), byteorder='big')
            # number of rows * number of columns = number of pixels (bytes) per image
            for _ in range(self.image_count):
                # self.plot_next_image(f)
                # represent the image as a column vector, convert elements are to the range of [0, 1.] to avoid overflow
                images.append(np.array(bytearray(f.read(self.rows * self.cols)), dtype=np.uint8).transpose() / 255.)
                # images.append(np.array([bytearray(f.read(self.rows * self.cols))], dtype=np.uint8).transpose() / 255.)
        
        labels = []
        with open(label_file_path, 'rb') as f:
            # skip the magic number
            f.seek(4)
            self.label_count = int.from_bytes(f.read(4), byteorder='big')
            assert self.image_count == self.label_count, 'number of images does not match number of labels'
            for _ in range(self.label_count):
                label = np.zeros(10)
                # label = np.zeros((10, 1))
                label[int.from_bytes(f.read(1), byteorder='big')] = 1.
                labels.append(label)
        
        return (np.array(images), np.array(labels))
        # return [(image, label) for image, label in zip(images, labels)]


    def plot_next_image(self, f):
        """
        helper method which plots the next digit on the screen.
        used to verify if pixel data is read correctly
        """
        image = np.zeros((self.rows, self.cols))
        for i in range (self.rows):
            image[i] = np.array(bytearray(f.read(self.cols)), dtype=np.uint8)
        plt.imshow(image, interpolation='none')
        plt.show()

if __name__ == '__main__':
    p = Parser()
    inputs = p.parse('./data/train-images-idx3-ubyte', './data/train-labels-idx1-ubyte')
    # print(len(inputs))
