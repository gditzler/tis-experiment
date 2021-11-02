import numpy as np 
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

def NeuralNetwork(X, Y): 
    num_classes = len(np.unique(Y))
    
    Y_train = to_categorical(Y, num_classes)
    X_train = X

    # Set the input shape
    input_shape = (X.shape[1],)

    # Create the model
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Configure the model and start training
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, 
              epochs=1000, 
              batch_size=256, 
              verbose=0, 
              validation_split=0.2)

    return model 