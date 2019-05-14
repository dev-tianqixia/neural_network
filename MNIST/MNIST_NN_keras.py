import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import parser

def run(inputs):
    images, labels = inputs

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=784))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=SGD(lr=0.3, decay=1e-5, momentum=0.9),
        metrics=['accuracy']
    )

    model.fit(images[:50000], labels[:50000], epochs=10, batch_size=20)
    score = model.evaluate(images[50000:], labels[50000:], batch_size=20)
    print(score)


if __name__ == "__main__":
    p = parser.Parser()
    inputs = p.parse('./data/train-images-idx3-ubyte', './data/train-labels-idx1-ubyte')
    run(inputs)