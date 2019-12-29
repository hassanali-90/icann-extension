import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "TkAgg"
from Utils import getProjectPath
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout


# split a singular block of sensor data into a sequences of gestures
# split is decided based on the change of activity in the target signal
def split_gestures(targets, sensor_readings):
    split_index = 0
    for i in range(1, len(targets)):
        if targets[i] < targets[i-1]:
            yield targets[split_index:i], sensor_readings[split_index:i, :]
            split_index = i
    if split_index <= len(targets)-1:
        yield targets[split_index:], sensor_readings[split_index:i, :]


def get_data(file_path, normalize=True, noise_factor=1):
    # read data file
    data = np.load(os.path.join(getProjectPath(), file_path))
    fused = data['fused']
    gyro = data['gyro']
    acc = data['acc']
    sensor_readings = np.hstack((fused, gyro, acc))
    targets = data['targets']
    gestures = data['gestures']

    # normalize, useNormalized=2 in the paper
    if normalize:
        sensor_readings[0:3] = np.max(np.linalg.norm(sensor_readings[:, 0:3], None, 1))
        sensor_readings[3:6] = np.max(np.linalg.norm(sensor_readings[:, 3:6], None, 1))
        sensor_readings[6:9] = np.max(np.linalg.norm(sensor_readings[:, 6:9], None, 1))

    # apply noise, default is noiseFactor=1 as in the paper
    sensor_readings[:, 0:3] += np.random.normal(0, 0.05 * noise_factor, size=(len(sensor_readings), 3))
    sensor_readings[:, 3:6] += np.random.normal(0, 0.5 * noise_factor, size=(len(sensor_readings), 3))
    sensor_readings[:, 6:9] += np.random.normal(0, 1.25 * noise_factor, size=(len(sensor_readings), 3))

    # split sensor data (fused, gyro, acc) into sequences of individual gestures as attempted by participants
    # should result in 10 gestures per participant
    # gestures commonly framed into 60 timesteps window
    individ_motions = list()
    individ_gesture = list()
    for x, y in split_gestures(targets[:, 2], sensor_readings):
        y = y[-60:]
        paddded_t = np.zeros((60, 9))
        paddded_t[:np.shape(y)[0], :np.shape(y)[1]] = y
        individ_motions.append(paddded_t)
        individ_gesture.append(gestures[:10])

    return individ_motions, individ_gesture


# config data location and file names
data_dir = 'dataSets/'
train_files = ['nike', 'julian', 'nadja', 'line']
test_files = ['stephan']
gestures = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# get training data
train_y = []
train_X = []
for train_file in train_files:
    for gesture in gestures:
        file_path = data_dir + train_file + '_' + str(gesture) + '_' + 'fullSet.npz'
        x, y = get_data(file_path)
        train_X.extend(x)
        train_y.extend(y)

train_X = np.asarray(train_X, dtype=np.float32)
train_y = np.asarray(train_y, dtype=np.float32)
print(train_X.shape)
print(train_y.shape)

# get testing data
test_X = []
test_y = []
for test_file in test_files:
    for gesture in gestures:
        file_path = data_dir + test_file + '_' + str(gesture) + '_' + 'fullSet.npz'
        x, y = get_data(file_path)
        test_X.extend(x)
        test_y.extend(y)

test_X = np.asarray(test_X, dtype=np.float32)
test_y = np.asarray(test_y, dtype=np.float32)
print(test_X.shape)
print(test_y.shape)


# config model
verbose, epochs, batch_size = 2, 50, 64
timesteps, features, output = train_X.shape[1], train_X.shape[2], train_y.shape[1]
model = Sequential()
model.add(GRU(100, input_shape=(timesteps, features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(output, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# train and evaluate model
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                    validation_data=(test_X, test_y), verbose=verbose)
_, accuracy = model.evaluate(test_X, test_y, batch_size=batch_size, verbose=0)

print("Model Accuracy: %.2f%%" % (accuracy*100))
# plot loss, val_loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
