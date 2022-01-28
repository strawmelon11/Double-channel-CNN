from __future__ import absolute_import, division, print_function, unicode_literals
from keras import backend, Model
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, multiply, AveragePooling2D
from keras.layers import Flatten, Input, MaxPooling2D, concatenate, Dense, Dropout
import keras
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np

tf.keras.backend.set_image_data_format('channels_last')
os.environ['KMP_WARNINGS'] = 'off'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(7)

# 数据集划分6:2:2

batchsize = 206
learning_rate = 0.005
##############
test_batchsize = 103
epochs = 250
model_png = ''
model_save = ''


file_in = ''  # 输入文件的存放目录
filename3 = ''  # 用于 归一化还原
file_in_pre = ''
file_out_pre = ''

# 训练集
feature_2_train = np.load(file_in + 'feature_2_train.npy')
feature_1_train = np.load(file_in + 'feature_1_train.npy')
cij_train = np.load(file_in + 'cij_train.npy')
road_dense_train = np.load(file_in + 'road_dense_train.npy')
y_train = np.load(file_in + 'out_train.npy')
# 测试集
feature_2_test = np.load(file_in + 'feature_2_test.npy')
feature_1_test = np.load(file_in + 'feature_1_test.npy')
cij_test = np.load(file_in + 'cij_test.npy')
road_dense_test = np.load(file_in + 'road_dense_test.npy')
y_test = np.load(file_in + 'out_test.npy')
# 验证集
feature_2_val = np.load(file_in + 'feature_2_val.npy')
feature_1_val = np.load(file_in + 'feature_1_val.npy')
cij_val = np.load(file_in + 'cij_val.npy')
road_dense_val = np.load(file_in + 'road_dense_val.npy')
y_val = np.load(file_in + 'out_val.npy')
print(feature_2_train.shape, feature_2_test.shape, feature_2_val.shape)

# 加载待预测数据
feature_2 = np.load(file_in_pre + 'feature_2.npy')
feature_1 = np.load(file_in_pre + 'feature_1.npy')
cij = np.load(file_in_pre + 'cij.npy')
road_dense = np.load(file_in_pre + 'road_dense.npy')
print(feature_2.shape, feature_1.shape, cij.shape, road_dense.shape)


# 自定义loss为RMSE均方根误差
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def rmse0(y_true, y_pred):
    MSE = np.sum(np.power((y_true.reshape(-1, 1) - y_pred), 2)) / len(y_true)  # y_true由一维矩阵转为二维矩阵
    return np.math.sqrt(MSE)


# 计算R2决定系数
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))  # 残差平方和
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))  # 总离差平方和
    return 1 - SS_res / (SS_tot + K.epsilon())


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def create_model():
    # 构建模型 LeNet
    # channel 1 ：2-features：check-in、POIs

    input1 = Input(batch_shape=(batchsize, 16, 16, 2))
    conv1_1 = Conv2D(4, (3, 3), activation='relu',
                     kernel_initializer=tf.glorot_normal_initializer())(input1)
    pool1_1 = MaxPooling2D(pool_size=2, strides=(2, 2))(conv1_1)
    # conv1_2 = Conv2D(4, (3, 3), activation='relu')(pool1_1)
    conv1_2 = Conv2D(8, (3, 1), activation='relu',
                     kernel_initializer=tf.glorot_normal_initializer())(pool1_1)
    conv1_3 = Conv2D(8, (1, 3), activation='relu')(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=3, strides=(2, 2))(conv1_3)

    flat1 = Flatten()(pool1_2)

    # channel 2：1-feature : population
    input2 = Input(batch_shape=(batchsize, 16, 16, 1))
    conv2_1 = Conv2D(2, (5, 5), activation='relu',
                     kernel_initializer=tf.glorot_normal_initializer())(input2)
    pool2_1 = MaxPooling2D(pool_size=2)(conv2_1)
    conv2_2 = Conv2D(4, (3, 3), activation='relu',
                     kernel_initializer=tf.glorot_normal_initializer())(pool2_1)
    pool2_2 = AveragePooling2D(pool_size=2)(conv2_2)
    # pool2_2 = AveragePooling2D()(conv2_2)
    # pool2_1 = MaxPooling2D(pool_size=2)(conv2_2)

    flat2 = Flatten()(pool2_2)

    # input3：1-feature:spatial competition index
    input3 = Input(batch_shape=(batchsize, 1))

    # input4：1-feature:road network density
    input4 = Input(batch_shape=(batchsize, 1))

    # merge
    merged = concatenate([flat1, flat2, input3, input4])

    drop = Dropout(0.1)(merged)
    dense1 = Dense(16, activation='relu')(drop)  # 14_1
    # dense2 = Dense(8, activation='relu')(dense1)  # 14_1
    outputs = Dense(1)(dense1)

    model = Model(inputs=[input1, input2, input3, input4], outputs=outputs)

    # 编译模型
    adam = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=[rmse, r_square])

    return model


model = create_model()
# 模型可视化
print(model.summary())

plot_model(model, show_shapes=True, to_file=model_png)

# 自适应调整学习率
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                              verbose=1, mode='auto', min_delta=0.00001, cooldown=0)

history = model.fit([feature_2_train, feature_1_train, cij_train, road_dense_train], y_train, epochs=epochs, batch_size=batchsize,
                    validation_data=([feature_2_test, feature_1_test, cij_test, road_dense_test], y_test), callbacks=[reduce_lr])


#  R^2
plt.figure(dpi = 300)
plt.plot(history.history['r_square'])
plt.plot(history.history['val_r_square'])
plt.title('DA_DCCNN_R^2: batchsize=' + str(batchsize) + '; lr=' + str(learning_rate))
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model/Metrics_Loss/DCCNN_R^2.png")
plt.show()
plt.savefig('R2.jpg', format='jpg', dpi=300)

#  LOSS
plt.figure(dpi = 300)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig("model/Metrics_Loss/DCCNN_Loss_formal.png")
plt.show()
plt.savefig('LOSS.jpg', format='jpg', dpi=300)

# ----------------------------------------------------------------------------
# Model 评估
#  model.predict模型预测,输入测试集,输出预测结果
pre_result = model.predict([feature_2_test, feature_1_test, cij_test, road_dense_train])
#  model.evaluate评估模型,不输出预测结果。 返回一个list，test_result[0]=loss,test_result[1]=rmse, test_result[2]=r_square
test_metrics = model.evaluate([feature_2_test, feature_1_test, cij_test, road_dense_test], y_test, batch_size=test_batchsize)
print("test_loss:", test_metrics[0])
# print("test_RMSE:", test_metrics[1])
print("test_R^2:", test_metrics[2])

# 待预测数据 格网
pre_grid = model.predict([feature_2, feature_1, cij, road_dense])

# 数据归一化还原
# 加载数据
data = pd.read_csv(filename3)
sales_max = data.loc[:, 'sales_sum'].max()
sales_min = data.loc[:, 'sales_sum'].min()

inverse_y_test = []  # 保存还原后的y_test
inverse_pre_result = []  # 保存还原后的pre_result
inverse_pre_grid = []  # 保存预测格网结果

for i in range(745):
    # 以万元为单位。除以10000
    inverse_pre_grid.append((pre_grid[i] * (sales_max - sales_min) + sales_min) / 10000)

for i in range(len(y_test)):
    # 以万元为单位。除以10000
    inverse_y_test.append((y_test[i] * (sales_max - sales_min) + sales_min) / 10000)
    inverse_pre_result.append((pre_result[i] * (sales_max - sales_min) + sales_min) / 10000)

inverse_y_test = np.array(inverse_y_test).ravel()  # 一维数组
inverse_pre_result = np.array(inverse_pre_result).ravel()
inverse_pre_grid = np.array(inverse_pre_grid).ravel()

RMSE = rmse0(y_test, pre_result)
RMSE1 = rmse0(inverse_y_test, inverse_pre_result)
MAE = np.sum(np.abs(y_test - pre_result.ravel())) / len(y_test)
MAPE=mape(y_test, pre_result)
print("test_MAE:", MAE)
print("test_RMSE:", RMSE)
print("test_MAPE:", MAPE)

############绘制折线图##########
plt.figure(dpi = 300)
plt.plot(np.arange(len(pre_result)), inverse_y_test, 'go-', label='true value')
plt.plot(np.arange(len(pre_result)), inverse_pre_result, 'ro-', label='predict value')
plt.title('DA_DCCNN  RMSE: %f' % RMSE + ': batchsize=' + str(batchsize) + '; lr=' + str(learning_rate))
plt.legend()  # 将样例显示出来
plt.show()
plt.savefig('line.jpg', format='jpg', dpi=300)
# 导出预测值
data = {'y_test': y_test.ravel(),
        'pre_result': pre_result.ravel(),
        'inverse_y_test(w)': inverse_y_test.ravel(),
        'inverse_pre_result(w)': inverse_pre_result.ravel()}

data_pre = {'result': pre_grid.ravel(),
            'inverse_result(w)': inverse_pre_grid.ravel()}

df = pd.DataFrame(data)
df.to_csv('.../out_file.csv', encoding='utf-8', index=False)
df_pre = pd.DataFrame(data_pre)
df_pre.to_csv(file_out_pre + 'pre_result.csv', encoding='utf-8', index=False)

model.save(model_save)
