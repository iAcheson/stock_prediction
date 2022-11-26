"""股票预测算法简单模版：仅供参考，有帮助给个star， ^_^ """

import os
import random
import pandas as pd
import mplfinance as mpf
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


def load_data(ktype):
    """载入股票数据：sh表示上证指数，貌似只能获取最近几天的数据"""
    data = ts.get_hist_data('sh', start='19990101', ktype=str(ktype)).reset_index().sort_values('date')
    file_ = 'data_hist.csv'
    # 本地更新积累历史数据
    if os.path.exists(file_):
        data = pd.read_csv(file_).append(data).drop_duplicates(subset=['date'], keep='first').sort_values('date')\
            .reset_index(drop=True)
    data.to_csv('data_hist.csv', index=False)
    # print(data)
    return data


def abnormal_detection(data):
    """异常值处理"""
    cols = data.columns
    for col in cols:
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


def data_scalar(x, y):
    """数据归一化"""
    scalar_x = MinMaxScaler(feature_range=(0, 1))
    scalar_y = MinMaxScaler(feature_range=(0, 1))
    x = scalar_x.fit_transform(x)
    y = scalar_y.fit_transform(y)
    return x, y, scalar_x, scalar_y


def data_set(x, y, lookback):
    """创建时间序列数据样本"""
    # 训练特征，训练标签，预测特征
    datax, datay, pred_x = [], [], []
    # print('x, y ---', x.shape, y.shape)
    for i in range(len(x) - lookback * 2 + 1):
        index = i + lookback
        datax.append(x[i: index])
        datay.append(y[index: index + lookback])
    pred_x.append(x[-lookback:])
    return np.array(datax), np.array(datay), np.array(pred_x)


def get_weekday(data, time_format='%Y-%m-%d %H:%M:%S'):
    """获取星期"""
    data['weekday'] = data['date'].map(lambda t: datetime.strptime(t, time_format).isoweekday())
    return data


def create_model(input_size, output_size):
    """构建模型"""
    inputs = tf.keras.Input(shape=(None, input_size))
    x = tf.keras.layers.Normalization()(inputs)
    x = tf.keras.layers.GRU(120, return_sequences=True, activation='tanh')(x)
    x = tf.keras.layers.Normalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GRU(60, return_sequences=False, activation='tanh')(x)
    x = tf.keras.layers.Normalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    y1 = tf.keras.layers.Dense(output_size)(x)
    y2 = tf.keras.layers.Dense(output_size)(x)
    y3 = tf.keras.layers.Dense(output_size)(x)
    y4 = tf.keras.layers.Dense(output_size)(x)
    y5 = tf.keras.layers.Dense(output_size)(x)
    y6 = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[y1, y2, y3, y4, y5, y6])
    # model = tf.keras.Model(inputs=[inputs], outputs=[y1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse',  # 损失函数用交叉熵
        metrics=["mse", 'mae']
    )
    model.summary()
    return model


def plot_learning_curves(history):
    """画训练曲线"""
    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.title('训练情况')
    plt.show()


def get_xticks(freq, lookback=48, time_format='%Y-%m-%d %H:%M:%S'):
    """获取时间横坐标"""
    today = str(datetime.now().date())
    start_time = today + ' 09:35'
    # time_format = '%Y-%m-%d %H:%M'
    print(start_time)
    date_time = datetime.strptime(start_time, time_format)
    xticks = []
    for i in range(lookback):
        time_ = str(date_time.time())[:-3]
        xticks.append(time_)
        if time_ == '11:30':
            date_time = date_time + timedelta(minutes=95)
        else:
            date_time = date_time + timedelta(minutes=freq)
    return xticks


def get_dates(freq, lookback=48):
    """获取时间索引"""
    today = str(datetime.now().date())
    start_time = today + ' 09:35'
    time_format = '%Y-%m-%d %H:%M'
    # print(start_time)
    date_time = datetime.strptime(start_time, time_format)
    xticks = []
    for i in range(lookback):
        time_ = str(date_time.time())[:-3]
        xticks.append(str(date_time))
        if time_ == '11:30':
            date_time = date_time + timedelta(minutes=95)
        else:
            date_time = date_time + timedelta(minutes=freq)
    return xticks


def plot_k_line(data):
    """画预测K线"""
    # 获取时间索引
    dates = get_dates(freqs)
    data = pd.DataFrame(data, columns=y_cols)
    data.index = list(map(lambda dat: datetime.strptime(dat, '%Y-%m-%d %H:%M:%S'), dates))
    # 修正预测值: 以收盘close为标准
    data['low'] = data[['close', 'low', 'high']].min(axis=1)
    data['high'] = data[['close', 'low', 'high']].max(axis=1)
    print(data)
    # 设置mplfinance的蜡烛颜色，up为阳线颜色，down为阴线颜色
    my_color = mpf.make_marketcolors(up='r',
                                     down='g',
                                     edge='inherit',
                                     wick='inherit',
                                     volume='inherit')
    # 设置图表的背景色
    my_style = mpf.make_mpf_style(marketcolors=my_color,
                                  # figcolor='(0.82, 0.83, 0.85)',
                                  # gridcolor='(0.82, 0.83, 0.85)'
                                  )
    # 增加5分钟收盘折线图
    ap = mpf.make_addplot(data[['close']])
    # 画整个k线图
    mpf.plot(data,
             addplot=ap,
             type='candle',
             style=my_style,
             volume=True)


def plot_prediction(freq, lookback, pred, labels, y_true=None):
    """画预测结果曲线图"""
    tomorrow = str((datetime.now() + timedelta(hours=24)).date())
    xticks = get_xticks(freq, lookback)
    x = range(lookback)
    plt.xlim(0, lookback)
    plt.plot([24, 24], [np.min(pred), np.max(pred)], 'm--')
    plt.plot(x, [np.mean(pred)] * lookback, 'm--')
    # 绘制多条曲线
    colors = ['green', 'red', 'b', 'c', 'y', 'm']
    if len(pred.shape) >= 2:
        # 绘制曲线间填充
        y1, y2 = pred[:, 0], pred[:, 1]
        plt.fill_between(x, y1, y2, where=(y1 > y2), color='C2', alpha=0.3, interpolate=True)
        plt.fill_between(x, y1, y2, where=(y1 <= y2), color='C3', alpha=0.3, interpolate=True)
        for i in range(pred.shape[-1]):
            plt.plot(x, pred[:, i], color=colors[i], label=labels[i])
    else:
        plt.plot(x, pred, color='red', label=labels[0])

    if y_true is not None:
        plt.ylim(np.min([pred, y_true]) - 1, np.max([pred, y_true]) + 1, 0.01)
        labels_true = [i + '_true' for i in labels]
        # print(y_true.shape)
        for i in range(y_true.shape[-1]):
            plt.plot(x, y_true[:, i], color=colors[2:][i], label=labels_true[i])
    else:
        plt.ylim(np.min(pred) - 1, np.max(pred) + 1, 0.01)

    plt.tight_layout()
    plt.xticks(x, xticks, color='black', rotation=60)
    plt.xlabel('time')
    plt.ylabel('index')
    plt.grid()
    plt.legend(loc=0, prop={'size': 18})
    plt.title(f'Index Forecast: {tomorrow}')
    plt.show()


if __name__ == '__main__':
    # 载入数据
    freqs = 5
    is_valid = False
    df = load_data(freqs)
    # 提取时间特征
    df = get_weekday(df)
    # 选择特征列及标签列
    x_cols = ['open', 'high', 'close', 'low', 'volume', 'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5',
              'v_ma10', 'v_ma20', 'weekday']
    y_cols = ['open', 'high', 'close', 'low', 'volume', 'ma5']
    append_cols = [i for i in y_cols if i not in x_cols]
    all_cols = x_cols + append_cols
    # 异常值处理
    # df = abnormal_detection(df[all_cols])
    # 归一化
    data_x, data_y = df[x_cols], df[y_cols]
    x, y, scalar_x, scalar_y = data_scalar(data_x, data_y)
    # 生成训练集
    steps = 4 * 60 // freqs
    _, traY, _ = data_set(x, y, steps)
    traX, traY1, pred_x = data_set(x, y[:, 0], steps)
    _, traY2, _ = data_set(x, y[:, 1], steps)
    _, traY3, _ = data_set(x, y[:, 2], steps)
    _, traY4, _ = data_set(x, y[:, 3], steps)
    _, traY5, _ = data_set(x, y[:, 4], steps)
    _, traY6, _ = data_set(x, y[:, 5], steps)

    # 模型训练
    clr = create_model(traX.shape[-1], traY1.shape[-1])
    if is_valid:
        hist = clr.fit(traX, [traY1, traY2, traY3, traY4, traY5, traY6], batch_size=64, epochs=20, validation_split=0.2, shuffle=False)
        plot_learning_curves(hist)
    hist = clr.fit(traX, [traY1, traY2, traY3, traY4, traY5, traY6], batch_size=64, epochs=20, shuffle=False)
    # 模型预测
    preds = clr.predict(pred_x)
    preds = np.concatenate(preds).T
    preds = scalar_y.inverse_transform(preds)
    # 画预测k线
    # print(preds)
    plot_k_line(preds)
