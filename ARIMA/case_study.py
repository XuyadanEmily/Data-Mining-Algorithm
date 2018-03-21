import tushare as ts
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA

def main():
    """
        引入数据，判断数据的平稳性，处理数据，建立模型
    :return:
    """
    #读取数据，股票数据接口
    ini_data = ts.get_hist_data('600848')
    index = pd.to_datetime(ini_data.index)
    sec_data  = pd.Series(ini_data['close'],index=index).sort_index()
    time_filed = sec_data.index[0] - sec_data.index[-1]
    print(time_filed.days)
    print(len(sec_data))

    #重采样：升采样
    third_data = sec_data.resample('D').interpolate('linear')
    data = third_data.truncate(before='20170101',after='20171231')
    print(data)
    print(len(data))
    #从图中观察
    data_rolling_mean = data.rolling(window=5).mean()
    data_rolling_mean.dropna(inplace=True)
    plt.figure(figsize=(12,8))
    plt.plot(data,color='black',label='Origin')
    plt.plot(data_rolling_mean,color='r',label='Rolling Mean')


    #看其ADF值
    # adfuller_value = adfuller(data)
    # print(adfuller_value)
    # result = pd.Series(adfuller_value[:4],index=['adf_result','p_value','numbers','use'])
    # for key,value in adfuller_value[4].items():
    #     result[key] = value
    # print('ADF result is:')
    # print(result)

    #ADF值没有想象中乐观，试一下rolling_mean的结果
    # adfuller_value = adfuller(data_rolling_mean)
    # print(adfuller_value)
    # result = pd.Series(adfuller_value[:4],index=['adf_result','p_value','numbers','use'])
    # for key,value in adfuller_value[4].items():
    #     result[key] = value
    # print('ADF result is:')
    # print(result)   #不理想

    data_diff = data.diff(1)
    data_diff.dropna(inplace=True)
    plt.plot(data_diff,color='y',label='data_diff')
    plt.legend(loc='best')
    plt.savefig('stock.png')
    plt.show()
    adfuller_value = adfuller(data_diff)
    print(adfuller_value)
    result = pd.Series(adfuller_value[:4],index=['adf_result','p_value','numbers','use'])
    for key,value in adfuller_value[4].items():
        result[key] = value
    print('ADF result is:')
    print(result)

    #找到其最佳的p/q值
    #先看其acf和pacf图
    f = plt.figure()
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)
    plot_acf(data_diff,lags=31,ax=ax1)
    plot_pacf(data_diff,lags=31,ax=ax2)
    plt.savefig('acf_pacf.png')
    plt.show()

    #找到最佳的p和q值
    # maxlags = 12
    # model_list = []
    # for p in range(1,maxlags+1):
    #     for q in range(1,maxlags+1):
    #         model = ARIMA(data,order=(p,1,q))
    #         try:
    #             result_model = model.fit()
    #         except:
    #             continue
    #         each_bic = result_model.bic
    #         model_list.append({'p':p,'q':q,'bic':each_bic})
    # bic = []
    # for i in model_list:
    #     bic.append(i['bic'])
    # index = bic.index(min(bic))
    # best_model = model_list[index]
    # best_p = best_model['p']
    # best_q = best_model['q']
    # best_bic = best_model['bic']
    # print('best_model:')
    # print('p:{};q:{},bic:{}'.format(best_p,best_q,best_bic))
    #运行之后找到模型的最佳值  p:1;q:1,bic:695.3648674307271，将其放入下列模型中运行

    # model = ARMA(data_diff,order=(1,5))
    # result_model = model.fit(disp=-1,method='css')

    #尝试用ARIMA模型
    model = ARIMA(data, order=(1,1,1))
    result_model = model.fit()

    start_date = datetime(2018,1,1)
    end_date = datetime(2018,1,5)

    # predict_result = result_model.predict(start_date,end_date,dynamic=True)
    predict_result = result_model.predict(start_date,end_date,dynamic=True)

    print('最初的预测结果：')
    print(predict_result)

    #将预测数据倒过去
    # data_recover = predict_result + data.shift(5)
    # data_recover.dropna(inplace=True)

    # plt.figure()
    # plt.plot(data,color='r')
    # plt.plot(data_recover,color='b')
    # plt.show()

    true_data = third_data.truncate(before='20180101',after='20180105')
    print('真实数据为：')
    print(true_data)

    #看结果模型与原来的数据的拟合情况



if __name__ == '__main__':
    main()

