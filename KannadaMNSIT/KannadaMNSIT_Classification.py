import os
import sys
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization


def classification_model(x_train,y_train):
	model = Sequential()
	
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())

	model.add(Flatten())
	model.add(Dense(24, activation='relu'))

	model.add(Dense(10, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


	model_summary=model.fit(x_train, y_train,epochs=13,batch_size=64,verbose=1,validation_split=0.20)

	return model

def main():
	train_data=pd.read_csv("./input/train.csv")
	test_data=pd.read_csv("./input/test.csv")
	samp_subm=pd.read_csv("./input/sample_submission.csv")
	
	x_train = train_data.copy()
	del x_train['label']
	
	y_train = train_data['label']
	y_train = to_categorical(y_train, num_classes = 10)
	
	x_test = test_data.copy()
	del x_test['id']
	
	x_train = x_train.values.reshape(-1,28,28,1)
	x_test = x_test.values.reshape(-1,28,28,1)

	x_train = x_train.astype('float32')/255
	x_test = x_test.astype('float32')/255

	model=classification_model(x_train,y_train)
	model.summary()

	y_test = model.predict(x_test)

	y_test_classes = np.argmax(y_test, axis = 1)
	output = pd.DataFrame({'id': samp_subm['id'],'label': y_test_classes})
	output.to_csv('submission.csv', index=False)
	
if __name__ == "__main__":
	main()
