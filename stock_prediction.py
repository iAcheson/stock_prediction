import random
import pandas as pd
import numpy as np
import tushare as ts
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


# 设置随机种子
SEED = 2022
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_data(cols, ktype):
    """载入股票数据：sh表示上证指数，貌似只能获取最近几天的数据"""
    # data = qs.get_data(['sh'], start='20221111', end=None, freq=5)
    data = ts.get_hist_data('sh', start='2020-11-11', ktype=str(ktype))
    return data[cols]


def abnormal_detection(data, col='close'):
    """异常值处理"""
    mean1 = data[col].quantile(q=0.25)  # 下四分位差
    mean2 = data[col].quantile(q=0.75)  # 上四分位差
    mean3 = mean2 - mean1  # 中位差
    topnum21 = mean2 + 1.5 * mean3
    bottomnum21 = mean2 - 1.5 * mean3
    print("正常值的范围：", topnum21, bottomnum21)
    print("是否存在超出正常范围的值：", any(data[col] > topnum21))
    print("是否存在小于正常范围的值：", any(data[col] < bottomnum21))
    # 检测存在超出正常范围的值，对该部分值进行替换
    replace_value = data[col][data[col] < topnum21].max()
    data.loc[data[col] > topnum21, col] = replace_value
    return data


def data_scalar(data):
    """数据归一化"""
    scalar = MinMaxScaler()
    data = scalar.fit_transform(data)
    return data, scalar


def data_split(data):
    """数据划分"""
    dataset_st = data
    # 划分训练集和测试集
    train_size = int(len(dataset_st) * 0.7)
    train, test = dataset_st[0: train_size], dataset_st[train_size: len(dataset_st)]
    return train, test


def data_set(dataset, lookback):
    """创建时间序列数据样本"""
    datax, datay = [], []
    for i in range(len(dataset) - lookback * 2):
        datax.append(dataset[i: i + lookback])
        datay.append(dataset[i + lookback: i + lookback + lookback])
    print(len(datax), len(datay))
    return np.array(datax), np.array(datay)


def create_model(tra_x, tra_y):
    """构建模型"""
    model = tf.keras.Sequential([
         tf.keras.layers.LSTM(120, input_shape=(tra_x.shape[1], tra_x.shape[2]), return_sequences=True),
         tf.keras.layers.Dropout(0.2),
         tf.keras.layers.LSTM(60),
         tf.keras.layers.Dropout(0.2),
         tf.keras.layers.Dense(30, activation='relu'),
         tf.keras.layers.Dense(tra_y.shape[1])
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mean_squared_error',  # 损失函数用交叉熵
        metrics=["mse"]
    )
    return model


def plot_learning_curves(history):
    """画训练曲线"""
    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.title('训练情况')
    plt.show()


def get_xticks(freq, lookback=50):
    """获取时间横坐标"""
    today = str(datetime.now().date())
    start_time = today + ' 09:30'
    time_format = '%Y-%m-%d %H:%M'
    print(start_time)
    date_time = datetime.strptime(start_time, time_format)
    xticks = []
    for i in range(lookback + 2):
        time_ = str(date_time.time())[:-3]
        xticks.append(time_)
        if time_ == '11:30':
            date_time = date_time + timedelta(minutes=90)
        else:
            date_time = date_time + timedelta(minutes=freq)
    return xticks


def plot_prediction(freq, lookback, pred):
    """画预测结果曲线图"""
    tomorrow = str((datetime.now() + timedelta(hours=24)).date())
    xticks = get_xticks(freq, lookback)[:lookback]
    x = range(lookback)
    y = pred
    plt.xlim(0, lookback)
    plt.ylim(min(pred) - 1, max(pred) + 1, 0.01)
    plt.plot(x, y, color='red')
    plt.xticks(x, xticks, color='black', rotation=60)
    plt.xlabel('time')
    plt.ylabel('index')
    plt.grid()
    plt.title(f'Index Forecast: {tomorrow}')
    plt.show()


if __name__ == '__main__':
    # 载入数据
    freqs = 5
    df = load_data(['close'], freqs)
    print(df)
    # 异常值处理：todo show 箱线图
    df = abnormal_detection(df)
    # 归一化
    df, sr = data_scalar(df)
    # 数据划分
    ttrain, ttest = data_split(df)
    # 根据划分的训练集测试集生成需要的时间序列样本数据
    steps = 4 * 60 // freqs + 2
    traX, traY = data_set(ttrain, steps)
    tesX, tesY = data_set(ttest, steps)
    print('trianX:,trianY', traX.shape, traY.shape)
    print('tesx:,tesy', tesX.shape, tesY.shape)
    print(np.concatenate((traX, tesX)).shape)
    clr = create_model(traX, traY)
    # hist = clr.fit(traX, traY, batch_size=64, epochs=25, validation_data=(tesX, tesY), validation_freq=1)
    hist = clr.fit(np.concatenate((traX, tesX)), np.concatenate((traY, tesY)), batch_size=64, epochs=20, validation_freq=1)
    predict_value = sr.inverse_transform(clr.predict(tesY[-1:]))[0]
    # plot_learning_curves(hist)
    plot_prediction(freqs, steps, predict_value)
