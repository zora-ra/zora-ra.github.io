import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import seed
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
from keras.models import Sequential
from keras.layers import SimpleRNN, Dropout, Dense, Flatten, TimeDistributed,Bidirectional, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import warnings
warnings.filterwarnings("ignore")
# seed(0)


## load data
data = pd.read_csv("S&P500.csv")
# print(data.head())
# print(data.shape)
# data.info()
data = data.dropna()


## Description of data
# time
Date = data["Date"]
# close price
close_price = data.loc[:,["Close"]].values.reshape(-1,1)

# scale data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(close_price)

# split data
scaled_train, scaled_test = train_test_split(dataset, test_size=0.2, shuffle=False)


# plot of the dependent variable versus time
fig, ax = plt.subplots(1, 1, figsize=(20, 8))
ax.plot(Date, close_price)
plt.xlabel("Day")
plt.ylabel("Stock price")
ax.xaxis.set_major_locator(ticker.MultipleLocator(360))
plt.xticks(rotation = 45)
plt.title("Plot of the Stock Price versus time")
plt.show()


# plot of sacling data
f,ax = plt.subplots(figsize = (20,8))
plt.plot(Date[:len(scaled_train)], scaled_train)
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
ax.xaxis.set_major_locator(ticker.MultipleLocator(360))
plt.xticks(rotation = 45)
plt.title("Plot of the Scaled stock Price versus time")
plt.show()


# create a data structure with 10 timesteps and 1 output
X_train = []
Y_train = []
timesteps = 10
for i in range(timesteps, len(scaled_train)):
    X_train.append(scaled_train[i-timesteps:i, 0])
    Y_train.append(scaled_train[i, 0])
X_train, y_train = np.array(X_train), np.array(Y_train)
# reshape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# print("X:",X_train)
# print("X size:",X_train.size)
# print("Y:",y_train)
# print("Y size:",y_train.size)


X_test = []
Y_test = []
for i in range(timesteps, len(scaled_test)):
    X_test.append(scaled_test[i-timesteps:i, 0])
    Y_test.append(scaled_test[i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Basic RNN model
print("------------------------ Basic RNN ------------------------")
model1 = Sequential()
model1.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))
model1.add(Dropout(0.2))
model1.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
model1.add(Dropout(0.2))
model1.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
model1.add(Dropout(0.2))
model1.add(SimpleRNN(units = 50))
model1.add(Dropout(0.2))
model1.add(Dense(units = 1))

# print(model1.summary())

# Compile
model1.compile(optimizer = 'adam', loss = "mean_squared_error")

# Fit
history1 = model1.fit(X_train, y_train, epochs = 5, batch_size = 32, verbose=0)
loss1 = history1.history['loss']




# Predict
testPredict = model1.predict(X_test)
trainPredict = model1.predict(X_train)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
Y_train = scaler.inverse_transform([Y_train])
testPredict = scaler.inverse_transform(testPredict)
Y_test = scaler.inverse_transform([Y_test])


# calculate RMSE
trainScore1 = math.sqrt(mean_squared_error(Y_train[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore1))
testScore1 = math.sqrt(mean_squared_error(Y_test[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore1))


# Visualization
# shifting test predictions for plotting
trainPredictPlot = np.empty_like(close_price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[timesteps:len(trainPredict)+timesteps, :] = trainPredict

# shifting test
testPredictPlot = np.empty_like(close_price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(timesteps*2):len(close_price), :] = testPredict

# plot baseline and predictions
f,ax = plt.subplots(figsize = (20,8))
plt.plot(Date, close_price, label="Real stock price")
plt.plot(trainPredictPlot, label = "Predicted stock price (train)")
plt.plot(testPredictPlot, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(180))
plt.legend()
plt.xticks(rotation = 45)
plt.title("RNN model")
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
plt.show()

# plot prediction and true test set
f,ax = plt.subplots(figsize = (20,8))
plt.plot(Date[len(close_price)-len(testPredict):], close_price[len(close_price)-len(testPredict):], label="Real stock price")
plt.plot(testPredict, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
plt.legend()
plt.xticks(rotation = 45)
plt.title("Predicted stock price using RNN model")
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
plt.show()





# Univariate LSTM
close_price = data.loc[:,["Close"]].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(close_price)

scaled_train, scaled_test = train_test_split(dataset, test_size=0.2, shuffle=False)


timesteps = 10
X_train = []
Y_train = []
for i in range(len(scaled_train)-timesteps-1):
    a = scaled_train[i:(i+timesteps), 0]
    X_train.append(a)
    Y_train.append(scaled_train[i + timesteps, 0])
X_train = np.array(X_train)
Y_train = np.array(Y_train)


X_test = []
Y_test = []
for i in range(len(scaled_test)-timesteps-1):
    a = scaled_test[i:(i+timesteps), 0]
    X_test.append(a)
    Y_test.append(scaled_test[i + timesteps, 0])
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# model Vanilla LSTM
print("------------------------ Vanilla LSTM ------------------------")
model2 = Sequential()
model2.add(LSTM(50, input_shape=(1, timesteps))) # 10 lstm neuron(block)
model2.add(Dense(1))

model2.compile(loss='mean_squared_error', optimizer='adam')
history2 = model2.fit(X_train, Y_train, epochs=5, batch_size=1, verbose=0)
loss2 = history2.history['loss']

# print(model2.summary())

# predict
trainPredict = model2.predict(X_train)
testPredict = model2.predict(X_test)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
Y_train_inver = scaler.inverse_transform([Y_train])
testPredict = scaler.inverse_transform(testPredict)
Y_test_incver = scaler.inverse_transform([Y_test])

# calculate RMSE
trainScore2 = math.sqrt(mean_squared_error(Y_train_inver[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore2))
testScore2 = math.sqrt(mean_squared_error(Y_test_incver[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore2))


# shifting train
trainPredictPlot = np.empty_like(close_price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[timesteps:len(trainPredict)+timesteps, :] = trainPredict

# shifting test
testPredictPlot = np.empty_like(close_price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(timesteps*2)+1:len(close_price)-1, :] = testPredict

# plot baseline and predictions
f,ax = plt.subplots(figsize = (20,8))
plt.plot(close_price, label="Real stock price")
plt.plot(trainPredictPlot, label = "Predicted stock price (train)")
plt.plot(testPredictPlot, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(180))
plt.legend()
plt.xticks(rotation = 45)
plt.title("Predict stock price using Vanilla LSTM model")
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
plt.show()

# plot prediction and true test set
f,ax = plt.subplots(figsize = (20,8))
plt.plot(Date[len(close_price)-len(testPredict):], close_price[len(close_price)-len(testPredict):], label="Real stock price")
plt.plot(testPredict, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
plt.legend()
plt.xticks(rotation = 45)
plt.title("Predicted stock price using Vanilla LSTM model")
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
plt.show()





# stacked LSTM
print("------------------------ Stacked LSTM ------------------------")
model3 = Sequential()
model3.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1,timesteps)))
model3.add(LSTM(50, activation='relu'))
model3.add(Dense(1))

model3.compile(optimizer='adam', loss='mse')
history3 = model3.fit(X_train, Y_train, epochs=5, batch_size=1, verbose=0)
loss3 = history3.history['loss']
# print(model3.summary())

# predict
trainPredict = model3.predict(X_train)
testPredict = model3.predict(X_test)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
Y_train_inver = scaler.inverse_transform([Y_train])
testPredict = scaler.inverse_transform(testPredict)
Y_test_incver = scaler.inverse_transform([Y_test])

# calculate RMSE
trainScore3 = math.sqrt(mean_squared_error(Y_train_inver[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore3))
testScore3 = math.sqrt(mean_squared_error(Y_test_incver[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore3))

# shifting train
trainPredictPlot = np.empty_like(close_price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[timesteps:len(trainPredict)+timesteps, :] = trainPredict

# shifting test
testPredictPlot = np.empty_like(close_price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(timesteps*2)+1:len(close_price)-1, :] = testPredict

# plot baseline and predictions
f,ax = plt.subplots(figsize = (20,8))
plt.plot(close_price, label="Real stock price")
plt.plot(trainPredictPlot, label = "Predicted stock price (train)")
plt.plot(testPredictPlot, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(180))
plt.legend()
plt.xticks(rotation = 45)
plt.title("Predict stock price using Stacked LSTM model")
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
plt.show()

# plot prediction and true test set
f,ax = plt.subplots(figsize = (20,8))
plt.plot(Date[len(close_price)-len(testPredict):], close_price[len(close_price)-len(testPredict):], label="Real stock price")
plt.plot(testPredict, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
plt.legend()
plt.xticks(rotation = 45)
plt.title("Predicted stock price using Stacked LSTM model")
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
plt.show()





## Bidirectional LSTM
print("------------------------- Bidirectional LSTM -------------------------")
model4 = Sequential()
model4.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(1,timesteps)))
model4.add(Dense(1))
model4.compile(optimizer='adam', loss='mse')
history4 = model4.fit(X_train, Y_train, epochs=5, batch_size=1, verbose=0)
loss4 = history4.history['loss']

# print(model4.summary())

# predict
trainPredict = model4.predict(X_train)
testPredict = model4.predict(X_test)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
Y_train_inver = scaler.inverse_transform([Y_train])
testPredict = scaler.inverse_transform(testPredict)
Y_test_incver = scaler.inverse_transform([Y_test])

# calculate RMSE
trainScore4 = math.sqrt(mean_squared_error(Y_train_inver[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore4))
testScore4 = math.sqrt(mean_squared_error(Y_test_incver[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore4))

# shifting train
trainPredictPlot = np.empty_like(close_price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[timesteps:len(trainPredict)+timesteps, :] = trainPredict

# shifting test
testPredictPlot = np.empty_like(close_price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(timesteps*2)+1:len(close_price)-1, :] = testPredict

# plot baseline and predictions
f,ax = plt.subplots(figsize = (20,8))
plt.plot(close_price, label="Real stock price")
plt.plot(trainPredictPlot, label = "Predicted stock price (train)")
plt.plot(testPredictPlot, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(180))
plt.legend()
plt.xticks(rotation = 45)
plt.title("Predict stock price using Bidirectional LSTM model")
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
plt.show()

# plot prediction and true test set
f,ax = plt.subplots(figsize = (20,8))
plt.plot(Date[len(close_price)-len(testPredict):], close_price[len(close_price)-len(testPredict):], label="Real stock price")
plt.plot(testPredict, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
plt.legend()
plt.xticks(rotation = 45)
plt.title("Predicted stock price using Bidirectional LSTM model")
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
plt.show()




# CNN LSTM
print("------------------------- CNN-LSTM -------------------------")
timesteps = 10
X_train = []
Y_train = []
for i in range(len(scaled_train)-timesteps-1):
    a = scaled_train[i:(i+timesteps), 0]
    X_train.append(a)
    Y_train.append(scaled_train[i + timesteps, 0])
X_train = np.array(X_train)
Y_train = np.array(Y_train)


X_test = []
Y_test = []
for i in range(len(scaled_test)-timesteps-1):
    a = scaled_test[i:(i+timesteps), 0]
    X_test.append(a)
    Y_test.append(scaled_test[i + timesteps, 0])
X_test = np.array(X_test)
Y_test = np.array(Y_test)


X_train = np.reshape(X_train, (X_train.shape[0], 1, 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, 1, X_test.shape[1]))

# define model
model5 = Sequential()
model5.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,1, timesteps)))
model5.add(TimeDistributed(MaxPooling1D(pool_size=2, padding='same')))
model5.add(TimeDistributed(Flatten()))
model5.add(LSTM(50, activation='relu'))
model5.add(Dense(1))
model5.compile(optimizer='adam', loss='mse')
model5.fit(X_train, Y_train, epochs=5,batch_size=1, verbose=0)
# print(model5.summary())


model5.compile(optimizer='adam', loss='mse')
# fit model
history5 = model5.fit(X_train, Y_train, epochs=5, verbose=0)
loss5 = history5.history['loss']


# predict
trainPredict = model5.predict(X_train)
testPredict = model5.predict(X_test)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
Y_train_inver = scaler.inverse_transform([Y_train])
testPredict = scaler.inverse_transform(testPredict)
Y_test_incver = scaler.inverse_transform([Y_test])


# calculate RMSE
trainScore5 = math.sqrt(mean_squared_error(Y_train_inver[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore5))
testScore5 = math.sqrt(mean_squared_error(Y_test_incver[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore5))


# shifting train
trainPredictPlot = np.empty_like(close_price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[timesteps:len(trainPredict)+timesteps, :] = trainPredict


# shifting test
testPredictPlot = np.empty_like(close_price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(timesteps*2)+1:len(close_price)-1, :] = testPredict


# plot baseline and predictions
f,ax = plt.subplots(figsize = (20,8))
plt.plot(close_price, label="Real stock price")
plt.plot(trainPredictPlot, label = "Predicted stock price (train)")
plt.plot(testPredictPlot, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(180))
plt.legend()
plt.xticks(rotation = 45)
plt.title("Predict stock price using CNN LSTM model")
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
plt.show()

# plot prediction and true test set
f,ax = plt.subplots(figsize = (20,8))
plt.plot(Date[len(close_price)-len(testPredict):], close_price[len(close_price)-len(testPredict):], label="Real stock price")
plt.plot(testPredict, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
plt.legend()
plt.xticks(rotation = 45)
plt.title("Predicted stock price using CNN LSTM model")
plt.xlabel("Day")
plt.ylabel("Scaled stock price")
plt.show()


# plot of loss comparison

f,ax = plt.subplots(figsize = (12,8))
plt.plot(loss1, label="RNN")
plt.plot(loss2, label="Vanilla LSTM")
plt.plot(loss3, label="Stacked LSTM")
plt.plot(loss4, label="Bidirectional LSTM")
plt.plot(loss5, label="CNN LSTM")
plt.legend()
plt.title("plot of loss comparison")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


f,ax = plt.subplots(figsize = (12,8))
# plt.plot(loss1, label="RNN")
plt.plot(loss2, label="Vanilla LSTM")
plt.plot(loss3, label="Stacked LSTM")
plt.plot(loss4, label="Bidirectional LSTM")
plt.plot(loss5, label="CNN LSTM")
plt.legend()
plt.title("plot of loss comparison")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
