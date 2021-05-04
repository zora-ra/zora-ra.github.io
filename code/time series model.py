import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from numpy.linalg import norm
from scipy.stats import chi2
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, q_stat
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import matplotlib.ticker as ticker
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
import warnings
warnings.filterwarnings("ignore")


# helper function
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))


def calculate_autocorrelation(data, lags):
    mean = np.mean(data)
    data_with_lag = data[lags:]
    data_without_lag = data[:len(data_with_lag)]
    numerater = np.sum((data_without_lag - mean) * (data_with_lag - mean))
    denominator = np.sum((data - mean) ** 2)
    autocorr = numerater / denominator
    return autocorr


def correlation_coefficient_cal(x, y):
    x_avg = sum(x) / len(x)
    y_avg = sum(y) / len(y)
    cov_xy = sum([(i - x_avg) * (j - y_avg) for i, j in zip(x, y)])
    sq = math.sqrt(sum([(i - x_avg) ** 2 for i in x]) * sum([(j - y_avg) ** 2 for j in y]))
    r = cov_xy / sq
    return r


# def Plot_acf(data, lags, method):
#     result = [calculate_autocorrelation(data, lag) for lag in range(0, lags + 1)]
#     result = result[:0:-1] + result
#     plt.figure(figsize=(12, 8))
#     plt.stem(np.arange(-lags, lags + 1, 1), result, use_line_collection=True, linefmt='-')
#     plt.xlabel("lags")
#     plt.ylabel("autocorrelation")
#     plt.title("ACF of {}".format(method))
#     plt.show()

def chi_square_test(y_train, error, lags):
    acf = []
    for i in range(lags):
        acf.append(calculate_autocorrelation(error, i + 1))
    acf.remove(acf[0])
    acf1 = np.array(acf)
    Q = len(error) * np.sum(acf1 ** 2)
    DOF = len(y_train) / 2 - 2
    alfa = 0.01
    chi_critical = chi2.ppf(1 - alfa, DOF)
    if Q < chi_critical:
        print("the residual is white")
    else:
        print("the residual is not white")


def get_Q(error, lags, method):
    acf = []
    for i in range(lags):
        acf.append(calculate_autocorrelation(error, i + 1))
    acf.remove(acf[0])
    acf1 = np.array(acf)
    Q = len(error) * np.sum(acf1 ** 2)
    print("the Q value of {} is:".format(method), round(Q, 2))


def get_errors(error, method):
    RMSE = np.sqrt(np.square(error).mean())
    Var = np.var(error)
    Mean = np.mean(error)
    print("the RMSE of {} is:".format(method), round(RMSE, 2))
    print("the variance {} is:".format(method), round(Var, 2))
    print("the mean of {} is:".format(method), round(Mean, 2))


def cal_PAC(ry, j, k):
    numerator = np.zeros((k, k))
    denominator = np.zeros((k, k))
    for r in range(k):
        for c in range(k):
            if k == 1:
                numerator[r][c] = ry[j + r + 1]
                denominator[r][c] = ry[np.abs(j - k + r + 1)]
            else:
                numerator[r][c] = ry[np.abs(r - c + j)]
                numerator[r][-1] = ry[j + r + 1]
                denominator[r][c] = ry[np.abs(r - c + j)]
                denominator[r][-1] = ry[np.abs(j - k + r + 1)]
    if np.linalg.det(denominator) == 0:
        return float('inf')
    else:
        return round((np.linalg.det(numerator) / np.linalg.det(denominator)), 3)


def GPAC_table(ry, row, col):
    GPAC = []
    for j in range(row):
        for k in range(1, col + 1):
            GPAC.append(cal_PAC(ry, j, k))
    GPAC_array = np.array(GPAC).reshape(row, col)
    GPAC_table = pd.DataFrame(GPAC_array, columns=np.arange(1, col + 1))
    plt.figure(figsize=(8, 5))
    sns.heatmap(GPAC_array, annot=True, xticklabels=np.arange(1, col + 1))
    plt.title("Generalized Partial Autocorrelation(GPAC) table")
    plt.xlabel("na")
    plt.ylabel("nb")
    plt.show()
    return GPAC_table


data = pd.read_csv("S&P500.csv")
# print(data.head())
# data.info()
data = data.dropna()



#  dependent variable
Y = data["Close"]
# time
Date = data["Date"]
print(len(Date))


# independent variable
X = data[["Open", "High", "Low", "Volume"]]
Open = data["Open"]
High = data["High"]
Low = data["Low"]
Volume = data["Volume"]


# acf of the dependent variable
plot_acf(Y.values, lags=50,alpha=.05,title="Autocorrelation of close price")
plot_pacf(Y.values, lags=50,alpha=.05,title="Partial autocorrelation of close price")
plt.show()

# correlation matrix
corr = data.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Correlation Matrix')
plt.show()


# split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
date_train, date_test = train_test_split(Date, test_size=0.2, shuffle=False)


# Stationary
print(ADF_Cal(y_train))


# differencing
order = 1
diff_y_train = np.diff(y_train,order)
print("first order difference:")
print(ADF_Cal(diff_y_train))

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(date_train[order:], diff_y_train)
plt.xlabel("Day")
plt.ylabel("Close price")
ax.xaxis.set_major_locator(ticker.MultipleLocator(360))
plt.xticks(rotation = 45)
plt.title("Close price with first order differencing")
plt.show()

plot_acf(diff_y_train, lags=50,alpha=.05,title="Autocorrelation of differenced close price")
plot_pacf(diff_y_train, lags=50,alpha=.05,title="Partial autocorrelation of differenced close price")
plt.show()


# decomposition

# STL
from statsmodels.tsa.seasonal import STL
res = STL(y_train.values, period=365).fit()
T = res.trend
S = res.seasonal
R = res.resid
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(date_train, T, label="trend")
ax.plot(date_train, S, label="seasonality")
plt.plot(date_train, R, label="reminder")
plt.xlabel("time")
plt.ylabel("close price")
ax.xaxis.set_major_locator(ticker.MultipleLocator(360))
plt.xticks(rotation = 45)
plt.title("STL decompostition")
plt.legend(loc=1)
plt.show()

# plot seasonal adjusted data
adjusted_data = y_train.values.reshape(len(y_train),) - S
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(date_train, y_train.values, label="original data")
ax.plot(date_train, adjusted_data, label="seasonally adjusted data")
plt.xlabel("time")
plt.ylabel("close price")
ax.xaxis.set_major_locator(ticker.MultipleLocator(360))
plt.xticks(rotation = 45)
plt.title("Seasonally adjusted data")
plt.legend(loc=1)
plt.show()

# plot trend adjusted data
adjusted_data = y_train.values.reshape(len(y_train),) - T
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(date_train, y_train.values, label="original data")
ax.plot(date_train, adjusted_data, label="trend adjusted data")
plt.xlabel("time")
plt.ylabel("close price")
ax.xaxis.set_major_locator(ticker.MultipleLocator(360))
plt.xticks(rotation = 45)
plt.title("Trend adjusted data")
plt.legend(loc=1)
plt.show()

Ft = max(0, 1-np.var(R)/np.var(T+R))
Fs = max(0, 1-np.var(R)/np.var(S+R))
print("The strength of trend is", round(Ft,2))
print("The strength of seasonality is", round(Fs,2))






# ARIMA

# # GPAC estimate orders
# # estimate orders using stationary data
# acf = []
# for i in range(50):
#     acf.append(calculate_autocorrelation(diff_y_train, i + 1))
# ry = [np.var(diff_y_train)]
# for i in range(49):
#     ry.append(acf[i + 1] * np.var(diff_y_train))
# GPAC_table(ry, 8, 8)
from pmdarima import auto_arima

model = auto_arima(y_train, trace=True, error_action="ignore",suppress_warnings=True)
model.fit(y_train)
forecast = model.predict(n_periods=len(y_test))

# plot prediction and true test set
f,ax = plt.subplots(figsize = (20,8))
plt.plot(Date[len(Y)-len(y_test):],y_test.values,label="Real stock price")
plt.plot(forecast, label = "Predicted stock price (test)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
plt.legend()
plt.xticks(rotation = 45)
plt.title("Predicted stock price using ARIMA model")
plt.show()