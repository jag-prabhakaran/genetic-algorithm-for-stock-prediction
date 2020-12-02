import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import talib
import random

# Importing dataset
dataset = pd.read_csv('') # Import data set stored on computer
dataset = dataset.dropna() # Drop missing values from data set
dataset = dataset[['Open', 'High', 'Low', 'Close']] # No date, adjusted close, volume data

# Preparing the dataset
# High - low price, close - open price, 3-day moving average, 10-day moving average, 30-day moving average, standard deviation for 5 days, relative strength index, Williams %R
dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['3day MA'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10day MA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['30day MA'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_dev']= dataset['Close'].rolling(5).std()
dataset['RSI'] = talib.RSI(dataset['Close'].values, timeperiod = 9)
dataset['Williams %R'] = talib.WILLR(dataset['High'].values, dataset['Low'].values, dataset['Close'].values, 7)

dataset['Price_Rise'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0) # Stores 1 when future closing price is greater

dataset = dataset.dropna() # Drops rows with NaN values

data = dataset.iloc[:, 4:] # Remove OHLC data and keep input features and output in new data frame

# Dimensions of data
n = data.shape[0]
p = data.shape[1]

# Make data a np.array
data = data.values

# Split the dataset into 80% training and 20% test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Standardize the data to avoid bias
scaler = MinMaxScaler(feature_range=(-1, 1)) # Scale inputs and targets
scaler.fit(data_train) # Fit to data sets
data_train = scaler.transform(data_train) # Transform function train data set
data_test = scaler.transform(data_test) # Transform function test data set

# Split the train and test data
X_train = data_train[:, 0:-1] # Predictor from training data
y_train = data_train[:, -1] # Target from training data
X_test = data_test[:, 0:-1] # Predictor from testing data
y_test = data_test[:, -1]# Target from testing data

# BULIDING THE ANN

n_features = X_train.shape[1] # Number of features in training data

# Nuerons - hidden layer
n_neurons_1 = 512 
n_neurons_2 = 256
n_neurons_3 = 128

net = tf.InteractiveSession() 

X = tf.placeholder(dtype=tf.float32, shape=[None, n_features]) # Placeholder for network's inputs
Y = tf.placeholder(dtype=tf.float32, shape=[None]) # Placeholder for network's outputs

# Intialize for 2 variable weights and bias 
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden Weights: each layer passes its output to the next layer as input
# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_features, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

# Output Weights
# Output layer
W_out = tf.Variable(weight_initializer([n_neurons_3, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden Layer - to to specify network topology and architecture
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

out = tf.transpose(tf.add(tf.matmul(hidden_3, W_out), bias_out)) # Ouput layer

mse = tf.reduce_mean(tf.squared_difference(out, Y)) # Cost function for optimization - generates a measure of deviation between networks predictions and observed training targets

opt = tf.train.AdamOptimizer().minimize(mse) # Optimizer - does necessary computation to adapt network's weight and bias variables during training

net.run(tf.global_variables_initializer())

# Fitting the NN
batch_size = 100
epochs = 10
# Run
for e in range(epochs):
    shuffle_data = np.random.permutation(np.arange(len(y_train))) # Shuffle training data
    X_train = X_train[shuffle_data] # Store input and target data give them to Network as inputs and targets
    y_train = y_train[shuffle_data] # Store input and target data give them to Network as inputs and targets
    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size] # Sample X data batch flows through network to output layer
        batch_y = y_train[start:start + batch_size] # Compares prediction to targets of Y batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y}) # Run optimzer with batch and update network parameters

# Predict method
pred = net.run(out, feed_dict={X: X_test}) # Pass X_Test as argument and store result in pred
y_pred = pred[0] # Convert pred data into data frame stored in y_pred
y_pred = pred[0] > 0.5 # Convert y_pred to store binary values

dataset['y_pred'] = np.NaN # Create new column in dataframe dataset storing NaN values
dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred # Slice dataframe to store values of ypred in the column
trade_dataset = dataset.dropna() # Drop all NaN values from dataset and store them in new dataframe

trade_dataset['Tomorrows Returns'] = 0. # Create new column in trade_dataset and intialized with value of 0 <- floating-point
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1)) # Store closing price of today / yesterday 
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1) # Shift values up by 1 so tmmrw values are stored against todays

trade_dataset['Strategy Returns'] = 0. # Create new column intialized with value 0 <- floating-point
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == True, # Store value in Tomorrow's Returns if value in ypred column stored True
             trade_dataset['Tomorrows Returns'], - trade_dataset['Tomorrows Returns']) # Else store negative of the vlue in Tomorrow's Returns in Strategy Returns

# Compute the cumulative returns for the market and the strategy
trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

# Plot the returns
plt.figure(figsize=(10,5))
plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()
