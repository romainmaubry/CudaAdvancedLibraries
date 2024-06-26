from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

#Get model data
(X_train,y_train),(X_test,y_test)=cifar10.load_data()

#Normalize data
X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255

#One hot encoding of the data
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

#Architecture of the CNN
model =Sequential()

#Add layers
model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

#Print model summary
model.summary()

#Train the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_test,y_test))

#Evaluate the model
score=model.evaluate(X_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

#Create figure
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('cnn.png')
