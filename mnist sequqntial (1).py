#!/usr/bin/env python
# coding: utf-8





# In[13]:

import tensorflow
import keras
from tensorflow.keras import Sequential
from keras.layers import Flatten, Dropout,Dense,Activation
from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


# In[2]:


(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[3]:


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[4]:


# Reshape data
X_train /= 255
X_test /= 255


# In[5]:


X_train.shape , X_test.shape


# In[6]:


from keras.utils import np_utils


# In[9]:


n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
Y_train.shape


# In[10]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[11]:


model= Sequential()


# In[14]:


model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))


# In[15]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[16]:


model.summary()


# In[17]:


model.fit(X_train, Y_train,batch_size=128, epochs=20,verbose=2,validation_data=(X_test, Y_test))


# In[19]:


'''from keras.models import load_model 
model.save('D:\DL 1\MNIST.h5')


# In[20]:


model = load_model('D:\DL 1\MNIST.h5')
'''

# In[ ]:
model_json = model.to_json()
with open("D:\DL 1\model\MNISTJ.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("D:\DL 1\model\MNIST.h5")

#%%
json_file = open('D:\DL 1\model\MNISTJ.json','r')
loaded_model_json = json_file.read()
json_file.close()
#%%
# use Keras model_from_json to make a loaded model
#from keras import model_from_json 
#%%
loaded_model = tensorflow.keras.models.model_from_json(loaded_model_json)
#%%
# load weights into new model

loaded_model.load_weights("D:\DL 1\model\MNIST.h5")
print("Loaded Model from disk")
#%%
# compile and evaluate loaded model

loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
