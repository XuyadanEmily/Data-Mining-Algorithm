import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA
import sys

date_parser = lambda date:pd.datetime.strptime(date,'%Y-%m')
data = pd.read_csv('/Users/xuyadan/Data_Analysis/Algorithms/ARIMA/AirPassengers.csv',
                   parse_dates=['Month'],date_parser=date_parser,index_col='Month')
ts = data['#Passengers']
ts_log = np.log(ts)
ts_log_rolling_mean = ts_log.rolling(window=12).mean()
ts_log_rolling_mean.dropna(inplace=True)
ts_log_mean_diff = ts_log_rolling_mean.diff(1)
ts_log_mean_diff.dropna(inplace=True)

ts_log_diff2 = ts_log_mean_diff.diff(1)
ts_log_diff2.dropna(inplace=True)
result = adfuller(ts_log_diff2)
print(result)

#画自相关系数图和偏自相关
# f = plt.figure()
# ax1 = f.add_subplot(211)
# plot_acf(ts_log_diff2,lags=31,ax=ax1)
# ax2 = f.add_subplot(212)
# plot_pacf(ts_log_diff2,lags=31,ax=ax2)
# plt.show()

def get_best_model(ts,maxlag):
    model_list = []
    for p in np.arange(maxlag):
        for q in np.arange(maxlag):
            model = ARMA(ts,order=(p,q))
            try:
                result_ARMA = model.fit(disp=-1,method='css')
            except:
                continue
            bic = result_ARMA.bic
            each_model = {'p':p,'q':q,'bic':bic}
            model_list.append(each_model)
    bic_list = []
    for i in model_list:
        bic_list.append(i['bic'])
    min_bic = bic_list.index(min(bic_list))
    best_model = model_list[min_bic]
    print('p:{};q:{};bic:{}'.format(best_model['p'],best_model['q'],best_model['bic']))
    return best_model

#找出最佳模型
best_model = get_best_model(ts_log_diff2,13)
p = best_model['p']
q = best_model['q']

#模型拟合
model = ARMA(ts_log_diff2,order=(p,q))
result_model = model.fit(disp=-1,method='css')
plt.figure()
plt.plot(ts_log_diff2,color='b',label='Origin')
plt.plot(result_model.fittedvalues,color='r',label='Fit')
plt.title('error:{}'.format(sum((ts_log_mean_diff-result_model.fittedvalues)**2)))
plt.show()

#模型的预测
predict_ts = result_model.predict()
diff2_recover = predict_ts + ts_log_mean_diff.shift(1)
diff1_recover = diff2_recover + ts_log_rolling_mean.shift(1)
rolling_mean_recover = diff1_recover*12 - ts_log.rolling(window=11).sum().shift(1)
log_recover = np.exp(rolling_mean_recover)
log_recover.dropna(inplace=True)


plt.figure()
plt.plot(ts,label='Origin')
plt.plot(log_recover,label='Pred')
plt.legend(loc='best')
plt.savefig('hhh.png')
plt.show()





#
#
# def test_stationarity(ts):
#     #画图，看均值（滑动平均值和指数加权平均值）
#     data_rolling_mean = ts.rolling(window=12).mean()
#     data_weighted_rolling_mean = ts.ewm(halflife=12).mean()
#     plt.figure()
#     plt.plot(ts,color='b',label='Origin')
#     plt.plot(data_rolling_mean,color='r',label='rolling mean')
#     plt.plot(data_weighted_rolling_mean,color='y',label='wighted rolling mean')
#     plt.legend(loc='best')
#     plt.show()
#
#     #看ADF值
#     results = adfuller(ts)
#     print(results)
#     return results
#
# print('原始数据的稳定性情况：')
# result1 = test_stationarity(ts)
#
# #去除不稳定性
# ts_log = np.log(ts)
# test_stationarity(ts_log)
#
# #求差分
# ts_log_diff = ts_log.diff(12)
# ts_log_diff.dropna(inplace=True)
# test_stationarity(ts_log_diff)
#
# #此时d=2
# result2 = seasonal_decompose(ts_log)
# season = result2.seasonal
# trend = result2.trend
# residual = result2.resid
#
# plt.figure()
# plt.plot(season,label='season')
# plt.plot(trend,label='trend')
# plt.plot(residual,label='residual')
# plt.show()






