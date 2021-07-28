# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('sales.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:6]
y = dataset[:,6]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the keras model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# make class predictions with the model
dataset_test = loadtxt('sales_test.csv', delimiter=',')
Xtest = dataset_test[:,0:6]
ytest = dataset_test[:,6]
predictions = model.predict(Xtest)
# summarize the first 5 cases
for i in range(len(Xtest)):
	print("X=%s, Predicted=%s (expected %s)" % (Xtest[i].tolist(), predictions[i], ytest[i]))
