import datetime
import os
import time
import warnings
import numpy as np
from numpy import newaxis
from tensorflow.keras.layers import Dense, Activation, Dropout,LSTM

from tensorflow.keras.models import Sequential
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings




#seq_len为周期数，即每次使用多少天的数据来推测结果
def load_data2(filename, seq_len):
    #将csv文件载入为numpy一维数组
    my_matrix = np.loadtxt(open(filename,"rb"),delimiter=",",skiprows=0)


    sequence_length = seq_len + 1
    #创建一个空数组来存储数据集
    set1=np.empty((my_matrix.size-seq_len, sequence_length))
    for index in range(my_matrix.size - seq_len):
        #输入数据是以行为单位进行训练的，需要将提取的矩阵转置为行向量
        temp=my_matrix[index:index+sequence_length,].T
        #归一化，将数组元素转换为-1~1之间的小数，目的是为了加快计算速度，归一化只能针对每行数据分别进行归一，不能对整体数组归一。
        temp=temp/temp[0]-1
        set1[index,:]=temp

    #训练集数量
    n_train = round(0.9 * (my_matrix.size - seq_len))
    
    #print(n_train,set1[1])
    #测试集数量
    n_test=my_matrix.size - seq_len-n_train
    
    x_train=set1[0:n_train,0:seq_len]
    y_train=set1[0:n_train,-1]

    x_test=set1[n_train:,0:-1]
    y_test=set1[n_train:,-1]

    #此例为单因子模型，所以最后一个维度填1
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

    return [x_train, y_train, x_test, y_test]

    

def build_model(layers):
    model = Sequential()
    #inputshape=(50, 1): 当把这层作为某个模型的第一层时，需要用到该参数，输入矩阵形状：(n_samples, dim_input)，dim_input指影响输出的特征数，此例为单因子模型所以填1，当建立多因子模型时填因子个数
    #units=50: 隐藏层神经元的个数,输出长度为50的向量，因为每个时间步都输出一次，共50个时间步，所以最后得到50*50的序列，至于输出整个序列还是最后一个序列取决于参数return_sequences
    #return_sequences：默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出
    model.add(LSTM(units=50,input_shape=(layers[1], layers[0]),return_sequences=True))
    print('lstm层输出n个50*50的矩阵',model.output_shape)
    model.add(Dropout(0.2))
    print('Dropout输出',model.output_shape)

    #第一个参数指的是输出的维度，这和第一个LSTM的输出维度并不一样，这也是LSTM比较“随意”的地方
    #LSTM(100,return_sequences=False)
    model.add(LSTM(layers[2],return_sequences=False))
    print('第二个lstm层输出',model.output_shape)
    model.add(Dropout(0.2))
    print('Dropout输出',model.output_shape)

    #units=1
    model.add(Dense(units=layers[3]))
    print('全连接层输出',model.output_shape)

    #激活函数可以通过设置单独的激活层实现，也可以在构造层对象时通过传递activation参数实现
    model.add(Activation("linear"))

    start = time.time()
    
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs





X_train, y_train, X_test, y_test = load_data2('sp500.csv', 50)

'''
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_train[0,:,:],y_train[0])
print("\n")
print(X_train[1,:,:],y_train[1])
'''

model = build_model([1, 50, 100, 1])
model.fit(X_train,y_train,batch_size=512,nb_epoch=1,validation_split=0.05)
print("\n")


# 保存模型
model.save('model.h5')   # HDF5文件，pip install h5py

y_predict=model.predict(X_test,batch_size=512,verbose=1)

# 可视化
import matplotlib.pyplot as plt


plt.plot(y_test,label="true")
plt.plot(y_predict,label="predict")

#绘制标签栏
#loc参数设置标签的显示位置，'best'为自适应方式
#ncol设置列的数量，使显示扁平化，当要表示的标签特别多的时候会有用
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)

plt.show()

