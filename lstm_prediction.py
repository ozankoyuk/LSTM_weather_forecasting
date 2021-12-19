# %%
"""
Bu kod VBM 683 Makine Öğrenmesi Dersi için
Ozan Köyük (N20230337) tarafından yazılmıştır.
"""
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from numpy.random import seed
from numpy import array

from tensorflow import random 

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from datetime import datetime as dt
from matplotlib import dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential


def data_split(sequence, TIMESTAMP):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + TIMESTAMP
        if end_ix > len(sequence)-1:
            break
        # i to end_ix as input
        # end_ix as target output
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# location,datetime,temp,dew_point,humidity,wind,wind_speed,wind_gust,pressure,condition
# 1 hour of data contains 2 30mins data.
original_dataset = pd.read_csv('all_data.csv')

dataset = original_dataset[['datetime', 'temp']].copy()

# Convert Fahrenheit to Celcius: C=(F-32)*(5/9)
dataset.temp = dataset.temp.apply(lambda x: round((x-32)*(5.0/9.0), 1))

# Perform medial filter
dataset['temp'] = medfilt(dataset['temp'], 3)

# Apply gaussian filter with sigma=1.2
dataset['temp'] = gaussian_filter1d(dataset['temp'], 1.2)

next_24_hours = dataset['datetime'].iloc[-48:]
_24_hours = [
    dt.strptime(_hr, '%Y-%m-%dT%H:%M:%S.000Z')
    for _hr in next_24_hours.to_list()
]

seed(1)
random.set_seed(1)
# Number of days to train from. %80 of the data will be used to train.
train_days = int(len(dataset)*0.8)

# Number of days to be predicted. %20 of the data will be used to test.
testing_days = len(dataset) - train_days

# Epoch -> one iteration over the entire dataset
N_EPOCHS = 1

# Batch_size -> divide dataset and pass into neural network.
BATCH_SIZE = 8

# Parse and divide data into size of 48 which is equals to 24 hour of data.
TIMESTAMP = 48

train_set = dataset[0:train_days].reset_index(drop=True)
test_set = dataset[train_days: train_days+testing_days].reset_index(drop=True)

# Get 'temp' column
"""
    temp
0   -2.801538
1   -2.817544
2   -2.908731
...
"""
training_set = train_set.iloc[:, 1:2].values
testing_set = test_set.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.fit_transform(testing_set)

X_train, y_train = data_split(training_set_scaled, TIMESTAMP)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

X_test, y_test = data_split(testing_set_scaled, TIMESTAMP)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Apply stacked LSTM wih 2 drop-outs
# activations: 'relu' / 'sigmoid' / 'tanh' / 'hardtanh' / 'leekly'
# return_sequences: True -> to pass results to the next iteration of LSTM
# input_shape: (X_train.shape[1], 1) -> shape is TIMESTAMP value
model = Sequential()
model.add(
    LSTM(
        100,
        activation='relu',
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
        )
    )
# Dropout: blocks random data for the given probability to next iteration.
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))

# Final iteration needs no return_sequences because its the final step.
model.add(LSTM(100))

# When return_sequences is set to False,
# Dense is applied to the last time step only.
model.add(Dense(1))

# Most used optimizers: adam, sgd, adadelta, adamax, nadam
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(
    X_train,
    y_train,
    epochs=N_EPOCHS,
    batch_size=BATCH_SIZE
)

# loss -> array of loss values for every iteration
# epochs -> count of epochs
loss = history.history['loss']
epochs = range(len(loss))

y_predicted = model.predict(X_test)

# convert predicted values into real values
y_predicted_descaled = sc.inverse_transform(y_predicted)
y_train_descaled = sc.inverse_transform(y_train)
y_test_descaled = sc.inverse_transform(y_test)

y_pred = y_predicted.ravel()
y_pred = [round(yx, 2) for yx in y_pred]

_min_predicted = int(min(y_predicted_descaled[-48:]))
_max_predicted = int(max(y_predicted_descaled[-48:]))

# Prediction graph
prediction_fig = plt.figure(1, figsize=(30, 20))
ax = plt.gca()
formatter = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(formatter)
plt.plot(
    _24_hours,
    dataset.temp.iloc[-48:],
    color='black', 
    linewidth=1, 
    label='True value'
)
plt.plot(
    _24_hours,
    y_predicted_descaled[-48:],
    color='red',
    label='Predicted Temperature'
)
plt.gcf().autofmt_xdate()
plt.xticks(_24_hours)
plt.yticks(range(_min_predicted-2, _max_predicted+2, 2))
plt.legend(frameon=False)
plt.ylabel("Temperature (C)")
plt.xlabel("Hours")
plt.title(f"Predicted data for {_24_hours[-1].strftime('%d-%m-%Y')}")
plt.grid()
plt.show()
plt.savefig(f"{N_EPOCHS}_PREDICTION.png")

# Epochs graph
epochs_plt = plt.figure(2, figsize=(15, 15))
plt.plot(epochs, loss, color='black')
plt.xticks(epochs)
plt.ylabel("Loss (MSE)")
plt.xlabel("Epoch")
plt.title("Training curve")
plt.grid()
epochs_plt.show()
epochs_plt.savefig(f"{N_EPOCHS}_EPOCHS.png")

# Find R^2 and mean squared errors
mse = mean_squared_error(dataset.temp.iloc[-48:], y_predicted_descaled[-48:])
r2 = r2_score(dataset.temp.iloc[-48:], y_predicted_descaled[-48:])
print("Mean Squared Error = " + str(round(mse, 2)))
print("R2 = " + str(round(r2, 2)))

# Open a file with access mode 'a'
file_object = open('RESULTS.txt', 'a')

# Append results at the end of file
file_object.write('\n================================')
file_object.write(f"\nEpochs     : {N_EPOCHS}")
file_object.write(f"\nBatch Size : {BATCH_SIZE}")
file_object.write(f"\nTimestamp  : {TIMESTAMP}")
file_object.write('\nMean Squared Error : ' + str(round(mse, 2)))
file_object.write('\nR^2 : ' + str(round(r2, 2)))

# Close the file
file_object.close()

"""                                                              
  ___                   _  __                 _    
 / _ \ ______ _ _ __   | |/ /___  _   _ _   _| | __
| | | |_  / _` | '_ \  | ' // _ \| | | | | | | |/ /
| |_| |/ / (_| | | | | | . \ (_) | |_| | |_| |   < 
 \___//___\__,_|_| |_| |_|\_\___/ \__, |\__,_|_|\_\
                                  |___/            
 _   _   ____     ___    ____    _____    ___    _____   _____   _____ 
| \ | | |___ \   / _ \  |___ \  |___ /   / _ \  |___ /  |___ /  |___  |
|  \| |   __) | | | | |   __) |   |_ \  | | | |   |_ \    |_ \     / / 
| |\  |  / __/  | |_| |  / __/   ___) | | |_| |  ___) |  ___) |   / /  
|_| \_| |_____|  \___/  |_____| |____/   \___/  |____/  |____/   /_/                                           
"""