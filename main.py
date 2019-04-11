import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

model_type = 'FNN'

model = None

if model_type is 'CNN':
	X_train = X_train.reshape(60000,28,28,1)
	X_test = X_test.reshape(10000,28,28,1)

	y_train = tf.one_hot(y_train, 10)
	y_test = tf.one_hot(y_test, 10)

	model = models.Sequential([
		Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
		Conv2D(16, kernel_size=3, activation='relu'),
		Flatten(),
		Dense(10, activation='softmax')
	])

	model.compile(
		optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)

else:
	X_train, X_test = X_train / 255, X_test / 255

	model = models.Sequential([
			Flatten(input_shape=(28, 28)),
			Dense(512, activation=tf.nn.relu),
			Dropout(0.3),
			Dense(512, activation=tf.nn.relu),
			Dropout(0.3),
			Dense(10, activation=tf.nn.softmax)
		])

	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)


model.fit(
		X_train,
		y_train,
		epochs=5,
		batch_size=64,
		steps_per_epoch=3
	)

model.evaluate(X_test, y_test)