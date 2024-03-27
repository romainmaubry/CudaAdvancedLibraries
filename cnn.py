from keras.datasets import cifar10
(X_train,y_train),(X_test,y_test)=cifar10.load_data()
X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255

from keras.utils import to_categorical

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)


#Architecture of the CNN
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

model =Sequential()

model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))


#Training

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_test,y_test))

score=model.evaluate(X_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

