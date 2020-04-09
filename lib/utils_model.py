import logging
import os
import csv
import sys
import random
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def prepare_train_valid_test_2d(data, test_size, valid_size):

    train_len = int(data.shape[0] * (1 - test_size - valid_size))
    valid_len = int(data.shape[0] * valid_size)

    train_set = data[0:train_len]
    valid_set = data[train_len: train_len + valid_len]
    test_set = data[train_len + valid_len:]

    return train_set, valid_set, test_set


def create_data(data, seq_len, input_dim, output_dim, horizon, verified_percentage):
    _data = data.copy()
    T = _data.shape[0]
    K = _data.shape[1]
    bm = binary_matrix(verified_percentage, T, K)
    _std = np.std(_data)
    _data[bm == 0] = np.random.uniform(_data[bm == 0] - _std, _data[bm == 0] + _std)
    # only take pm10 and pm2.5 to predict
    pm_data = _data[:, -output_dim:].copy()
    en_x = np.zeros(shape=((T - seq_len - horizon), seq_len, input_dim))
    de_x = np.zeros(shape=((T - seq_len - horizon), horizon, output_dim))
    de_y = np.zeros(shape=((T - seq_len - horizon), horizon, output_dim))

    for i in range(T - seq_len - horizon):
        en_x[i, :, :] = _data[i:i + seq_len]

        de_x[i, :, :] = pm_data[i + seq_len - 1:i + seq_len + horizon - 1]
        de_x[i, 0, :] = 0
        de_y[i, :, :] = pm_data[i + seq_len:i + seq_len + horizon]

    return en_x, de_x, de_y


def load_dataset(seq_len, horizon, input_dim, output_dim, dataset, test_size, valid_size, verified_percentage):
    raw_data = np.load(dataset)['monitoring_data']
    
    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, test_size=test_size, valid_size=valid_size)
    data = {}
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(train_data2d)
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)
    data['test_data_norm'] = test_data2d_norm.copy()

    encoder_input_train, decoder_input_train, decoder_target_train = create_data(train_data2d_norm,
                                                                                        seq_len=seq_len,
                                                                                        input_dim=input_dim,
                                                                                        output_dim=output_dim,
                                                                                        horizon=horizon,
                                                                                        verified_percentage=verified_percentage)
    encoder_input_val, decoder_input_val, decoder_target_val = create_data(valid_data2d_norm,
                                                                                seq_len=seq_len,
                                                                                input_dim=input_dim,
                                                                                output_dim=output_dim,
                                                                                horizon=horizon,
                                                                                verified_percentage=verified_percentage)
    
    encoder_input_eval, decoder_input_eval, decoder_target_eval = create_data(test_data2d_norm,
                                                                                        seq_len=seq_len,
                                                                                        input_dim=input_dim,
                                                                                        output_dim=output_dim,
                                                                                        horizon=horizon,
                                                                                        verified_percentage=verified_percentage)
    for cat in ["train", "val", "eval"]:
        e_x, d_x, d_y = locals()["encoder_input_" + cat], locals()["decoder_input_" + cat], locals()["decoder_target_" + cat]
        data["encoder_input_" + cat] = e_x
        data["decoder_input_" + cat] = d_x
        data["decoder_target_" + cat] = d_y
    
    data['scaler'] = scaler
    return data

def mae(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        # cal mse
        error_mae = mean_absolute_error(test_arr, prediction_arr)
        return error_mae

def cal_error(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        # cal mse
        error_mae = mean_absolute_error(test_arr, prediction_arr)

        # cal rmse
        error_mse = mean_squared_error(test_arr, prediction_arr)
        error_rmse = np.sqrt(error_mse)

        # cal mape
        y_true, y_pred = np.array(test_arr), np.array(prediction_arr)
        error_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        error_list = [error_mae, error_rmse, error_mape]
        print("MAE: %.4f" % (error_mae))
        print("RMSE: %.4f" % (error_rmse))
        print("MAPE: %.4f" % (error_mape))
        return error_list

def save_metrics(error_list, log_dir, alg):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    error_list.insert(0, dt_string)
    with open(log_dir + alg + "_metrics.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(error_list)

def binary_matrix(verified_percentage, row, col):
    tf = np.array([1, 0])
    bm = np.random.choice(tf, size=(row, col), p=[verified_percentage, 1.0 - verified_percentage])
    return bm
