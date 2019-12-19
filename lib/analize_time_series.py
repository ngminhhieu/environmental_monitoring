from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
series = read_csv('data/csv/full_original_data_mean.csv', header=0, index_col=0)
# series = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv', header=0, index_col=0)
pyplot.plot(series['PM2.5'])
pyplot.title('PM2.5', fontsize=16)
pyplot.show()
plot_acf(series['PM2.5'], lags=1440)
pyplot.show()
