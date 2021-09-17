from tensorflow.keras.layers import LSTM,Flatten,Dense,Dropout,InputLayer,Reshape,Bidirectional,Conv2D,MaxPool2D,BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam,SGD
def build_model():
    model = Sequential([
        InputLayer((8,10,1)),
        BatchNormalization(),
        Conv2D(32,(2,2),padding="same",input_shape=(8,10,1),kernel_initializer="he_normal"),
        Dropout(0.2),
        Flatten(),
        Dense(10,activation="leaky_relu",kernel_initializer="he_normal"),
        Dense(1,activation="sigmoid",name="output_layer"),
    ])
    model.compile(loss="binary_crossentropy",optimizer=Adam(0.0001),metrics=["accuracy"])
    return model

