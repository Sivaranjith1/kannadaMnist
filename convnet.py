# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

# %%
NAME = "Kannada-Mnist"

# %%
X = pickle.load(open('X.pickle', 'rb'))
Y = pickle.load(open('Y.pickle', 'rb'))

X[0]
# %%
X = X/255.0

model = Sequential()

model.add(Conv2D(64, (2, 2), input_shape=(28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

model.fit(X, Y, batch_size=40, validation_split=0.1, epochs=10)

# %%


# %%
