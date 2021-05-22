import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

model_type = 'CNN'

model = None

if model_type == 'CNN':
	X_train = X_train.reshape(60000,28,28,1)
	X_test = X_test.reshape(10000,28,28,1)

	y_train = tf.one_hot(y_train, 10)
	y_test = tf.one_hot(y_test, 10)

	model = models.Sequential([
		Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
		Conv2D(16, kernel_size=3, activation='relu'),
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


callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
    #TensorBoard(log_dir='./board_logs'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)
]

model.fit(
		X_train,
		y_train,
		validation_data=(X_test, y_test),
		epochs=5000,
		batch_size=32,
		steps_per_epoch=1024,
		callbacks=callbacks
	)

model.evaluate(X_test, y_test)