from pandas import read_csv
import constant
import seaborn as sns

target_feature = ['PM2.5']
dataset = 'data/csv/hanoi_data_median.csv'
dataset = read_csv(dataset, usecols=constant.hanoi_features+target_feature)
corr = dataset.corr(method='pearson')
print(corr)
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
