import pandas as pd
import numpy as np



"""output_fields là mảng string gồm đầu ra người dùng lựa chọn"""
def reconstruct_file(url_dataset, output_fields):
    dataset_pd = pd.read_csv(url_dataset)
    # chuyển các fields cần dự đoán về cuối cho dễ xử lý    
    for field in output_fields:
        drop_col = dataset_pd.pop(field)
        dataset_pd.insert(len(dataset_pd.columns), field, drop_col)
    
    # change datetime type to float
    if 'date' in dataset_pd.columns:
        dataType = dict(dataset_pd.dtypes)
        
        print(dataType)
        if dataType['date'] != 'str':
            dataset_pd['date'] = pd.to_datetime(dataset_pd['date'])
            dataset_pd['date'] = dataset_pd['date'].values.astype(float)
        
        # convert datetime to string 
        else:
            list_date = dataset_pd['date'].to_numpy()
            print(list)
            dates = np.zeros(shape=(len(dataset_pd['date']), 3))
            for i, date in enumerate(list_date):
                dates[i] = date.split('/', 2)
            
            if max(dates[:, 0]) == 31:
                dataset_pd.insert(0, 'day', dates[:, 0])
                dataset_pd.insert(0, 'month', dates[:, 1])
                
            elif max(dates[:, 0]) == 12:
                dataset_pd.insert(0, 'month', dates[:, 0])
                dataset_pd.insert(0, 'day', dates[:, 1])
            dataset_pd.insert(0, 'year', dates[:, 2])
    return dataset_pd

def split_data(dataset, train_per, valid_per, output_fields):
    dataset = reconstruct_file(dataset, output_fields)
    dataset = dataset.to_numpy()
    number_of_output_fields = len(output_fields)
    # split data into X and y
    X = dataset[:, 0:-number_of_output_fields]
    Y = dataset[:, -number_of_output_fields:]
    # split data into train and test sets
    train_size = int(len(dataset)*train_per)
    valid_size = int(len(dataset)*valid_per)
    X_train = X[0:train_size]
    y_train = Y[0:train_size]

    X_valid = X[train_size:train_size+valid_size]
    y_valid = Y[train_size:train_size+valid_size]
    
    X_test = X[train_size:train_size+valid_size:]
    y_test = Y[train_size:train_size+valid_size:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test