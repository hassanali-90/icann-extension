import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "TkAgg"
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.optimizers import Adam

from DataSet import UniHHIMUGestures

# get data using exact same logic as the paper
trainset = UniHHIMUGestures(dataDir='dataSets/',
                            train=True,
                            useNormalized=2,
                            learnTreshold=False)

testset = UniHHIMUGestures(dataDir='dataSets/',
                           train=False,
                           useNormalized=2,
                           learnTreshold=False)

train_X = []
train_y = []
test_X = []
test_y = []
for inputs, targets in trainset:
    train_X.extend(inputs)
    train_y.extend(targets)

for inputs, targets in testset:
    test_X.extend(inputs)
    test_y.extend(targets)

train_X = np.asarray(train_X, dtype=np.float32)
train_y = np.asarray(train_y, dtype=np.float32)
test_X = np.asarray(test_X, dtype=np.float32)
test_y = np.asarray(test_y, dtype=np.float32)

# shift labels one timestep
train_y[0, :] = 0
train_y = np.roll(train_y, -1, axis=0)
test_y[0, :] = 0
test_y = np.roll(test_y, -1, axis=0)

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# config model
verbose, epochs, batch_size = 2, 50, 64
timesteps, features, outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(timesteps, features)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(25))
model.add(Dense(100, activation='relu'))
model.add(Dense(outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.9), metrics=['accuracy'])
# train and evaluate model
history = model.fit(train_X, train_y, epochs=epochs, validation_data=(test_X, test_y), batch_size=batch_size, verbose=verbose)
_, accuracy = model.evaluate(test_X, test_y, batch_size=batch_size, verbose=verbose)
print("Model Accuracy: %.2f%%" % (accuracy*100))


# plot loss and val_loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

'''
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
'''