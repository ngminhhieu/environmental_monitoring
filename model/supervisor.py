import matplotlib
import os
import time
import tensorflow.keras.callbacks as keras_callbacks
import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from lib import utils_model
# from tensorflow.keras.utils import plot_model
from model.bilstm_ed_construction import bilstm_ed_model_construction
from model.lstm_ed_construction import lstm_ed_model_construction
from model.gru_ed_construction import gru_ed_model_construction
from datetime import datetime
import time
import matplotlib.pyplot as plt

class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class EncoderDecoder():

    def __init__(self, is_training=True, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        self._test_kwargs = kwargs.get('test')
        self._model_kwargs = kwargs.get('model')
        self._alg_name = self._kwargs.get('alg')

        # data args
        self._dataset = self._data_kwargs.get('dataset')
        self._test_size = self._data_kwargs.get('test_size')
        self._valid_size = self._data_kwargs.get('valid_size')
        self._test_batch_size = self._data_kwargs.get('test_batch_size')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        # log_level = self._kwargs.get('log_level', 'INFO')
        # self._logger = utils_model.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        # self._logger.info(kwargs)

        # Model's Args
        self._type = self._model_kwargs.get('type')
        self._rnn_units = self._model_kwargs.get('rnn_units')
        self._seq_len = self._model_kwargs.get('seq_len')
        self._horizon = self._model_kwargs.get('horizon')
        self._input_dim = self._model_kwargs.get('input_dim')
        self._output_dim = self._model_kwargs.get('output_dim')
        self._rnn_layers = self._model_kwargs.get('rnn_layers')
        self._verified_percentage = self._model_kwargs.get('verified_percentage')

        # Train's args
        self._dropout = self._train_kwargs.get('dropout')
        self._epochs = self._train_kwargs.get('epochs')
        self._batch_size = self._data_kwargs.get('batch_size')
        self._optimizer = self._train_kwargs.get('optimizer')

        # Test's args
        self._run_times = self._test_kwargs.get('run_times')

        # Load data
        self._data = utils_model.load_dataset(seq_len=self._seq_len, horizon=self._horizon,
                                        input_dim=self._input_dim, output_dim=self._output_dim,
                                        dataset=self._dataset,
                                        test_size=self._test_size, valid_size=self._valid_size,
                                        verified_percentage=self._verified_percentage)

        self.callbacks_list = []

        self._checkpoints = ModelCheckpoint(
            self._log_dir + "best_model.hdf5",
            monitor='val_loss', verbose=1,
            save_best_only=True,
            mode='auto', save_freq='epoch')
        self._earlystop = EarlyStopping(monitor='val_loss', patience=self._train_kwargs.get('patience'),
                                        verbose=1, mode='auto')
        self._time_callback = TimeHistory()
    
        self.callbacks_list.append(self._checkpoints)
        self.callbacks_list.append(self._earlystop)
        self.callbacks_list.append(self._time_callback)
                                                        
        if self._type == 'lstm_ed':
            if is_training:
                self.model = lstm_ed_model_construction(self._input_dim, self._output_dim, self._rnn_units, self._dropout,
                                                        self._optimizer, self._log_dir, is_training=is_training)
            else:
                self.model, self.encoder_model, self.decoder_model = lstm_ed_model_construction(self._input_dim, self._output_dim, self._rnn_units, self._dropout,
                                                        self._optimizer, self._log_dir, is_training=is_training)
        
        elif self._type == 'bilstm_ed':
            if is_training:
                self.model = bilstm_ed_model_construction(self._input_dim, self._output_dim, self._rnn_units, self._dropout,
                                                        self._optimizer, self._log_dir, is_training=is_training)
            else:
                self.model, self.encoder_model, self.decoder_model = bilstm_ed_model_construction(self._input_dim, self._output_dim, self._rnn_units, self._dropout,
                                                        self._optimizer, self._log_dir, is_training=is_training)

        else:
            raise RuntimeError("Model type is invalid!")

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            log_dir = kwargs.get('base_dir')
            # type_model = kwargs['model'].get('type')
            # batch_size = kwargs['data'].get('batch_size')
            # rnn_layers = kwargs['model'].get('rnn_layers')
            # rnn_units = kwargs['model'].get('rnn_units')
            # structure = '-'.join(
            #     ['%d' % rnn_units for _ in range(rnn_layers)])
            seq_len = kwargs['model'].get('seq_len')
            horizon = kwargs['model'].get('horizon')
            # input_dim = kwargs['model'].get('input_dim')
            # output_dim = kwargs['model'].get('output_dim')
            # verified_percentage = kwargs['model'].get('verified_percentage')

            run_id = '%d_%d/' % (seq_len, horizon)
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def train(self):
        self.model.compile(optimizer=self._optimizer, loss='mse', metrics=['mse', 'mae'])

        training_history = self.model.fit([self._data['encoder_input_train'], self._data['decoder_input_train']],
                                          self._data['decoder_target_train'],
                                          batch_size=self._batch_size,
                                          epochs=self._epochs,
                                          callbacks=self.callbacks_list,
                                          validation_data=([self._data['encoder_input_val'],
                                                            self._data['decoder_input_val']],
                                                           self._data['decoder_target_val']),
                                          shuffle=True,
                                          verbose=2)
        if training_history is not None:
            self._plot_training_history(training_history)
            self._save_model_history(training_history)
            config = dict(self._kwargs)
            config_filename = 'config.yaml'
            config['train']['log_dir'] = self._log_dir
            with open(os.path.join(self._log_dir, config_filename), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        return self._time_callback.times

    def test(self):
        scaler = self._data['scaler']
        start_time = start_time = time.time()
        data_test = self._data['test_data_norm'].copy()
        # this is the meterogical data
        other_features_data = data_test[:, 0:(self._input_dim-self._output_dim)].copy()
        pm_data = data_test[:, -self._output_dim:].copy()
        T = len(data_test)
        l = self._seq_len
        h = self._horizon
        bm = utils_model.binary_matrix(self._verified_percentage, len(data_test), pm_data.shape[1])
        pd = np.zeros(shape=(T, self._output_dim), dtype='float32')
        pd[:l] = pm_data[:l]
        _pd = np.zeros(shape=(T, self._output_dim), dtype='float32')
        _pd[:l] = pm_data[:l]
        iterator = tqdm(range(0, T - l - h, h))
        for i in iterator:
            if i+l+h > T-h:
                # trimm all zero lines
                pd = pd[~np.all(pd==0, axis=1)]
                _pd = _pd[~np.all(_pd==0, axis=1)]
                iterator.close()
                break
            input = np.zeros(shape=(self._test_batch_size, l, self._input_dim))
            input[0, :, :] = data_test[i:i+l].copy()
            yhats = self._predict(input)
            _pd[i + l:i + l + h] = yhats

            # update y
            _gt = pm_data[i + l:i + l + h].copy()
            _bm = bm[i + l:i + l + h].copy()
            pd[i + l:i + l + h] = yhats * (1.0 - _bm) + _gt * _bm
        
        
        inference_time = (time.time()-start_time)
        # rescale metrics
        residual_row = len(other_features_data)-len(_pd)
        if residual_row != 0:
            other_features_data = np.delete(other_features_data, np.s_[-residual_row:], axis=0)
        inverse_pred_data = scaler.inverse_transform(np.concatenate((other_features_data,_pd), axis=1))
        predicted_data = inverse_pred_data[:,-self._output_dim:]
        inverse_actual_data = scaler.inverse_transform(data_test[:predicted_data.shape[0]])
        ground_truth = inverse_actual_data[:, -self._output_dim:]
        np.save(self._log_dir+'pd', predicted_data)
        np.save(self._log_dir+'gt', ground_truth)
        # save metrics to log dir
        error_list = utils_model.cal_error(ground_truth.flatten(), predicted_data.flatten())
        error_list = error_list + [inference_time]
        mae = utils_model.mae(ground_truth.flatten(), predicted_data.flatten())
        utils_model.save_metrics(error_list, self._log_dir, self._alg_name)
        return mae

    def _predict(self, source):
        states_value = self.encoder_model.predict(source)
        target_seq = np.zeros((self._test_batch_size, 1, self._output_dim))
        preds = np.zeros(shape=(self._horizon, self._output_dim),
                        dtype='float32')
        for i in range(self._horizon):
            output = self.decoder_model.predict([target_seq] + states_value)
            yhat = output[0]
            # store prediction
            preds[i] = yhat
            # update target sequence
            target_seq = yhat
            # Update states
            states_value = output[1:]
        return preds

    def load(self):
        self.model.load_weights(self._log_dir + 'best_model.hdf5')

    def _save_model_history(self, model_history):
        loss = np.array(model_history.history['loss'])
        val_loss = np.array(model_history.history['val_loss'])
        dump_model_history = pd.DataFrame(index=range(loss.size),
                                          columns=['epoch', 'loss', 'val_loss', 'train_time'])

        # check time lai o day
        now = datetime.now()
        training_time = now.strftime("%d/%m/%Y %H:%M:%S")

        dump_model_history['epoch'] = range(loss.size)
        dump_model_history['time'] = training_time        
        dump_model_history['loss'] = loss
        dump_model_history['val_loss'] = val_loss

        if self._time_callback.times is not None:
            dump_model_history['train_time'] = self._time_callback.times

        dump_model_history.loc[-1, -1] = np.sum(self._time_callback.times)
        dump_model_history.to_csv(self._log_dir + 'training_history.csv', index=False)

    def _plot_training_history(self, model_history):
        plt.plot(model_history.history['loss'], label='loss')
        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + '[loss]{}.png'.format(self._alg_name))
        plt.legend()
        plt.close()

        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + '[val_loss]{}.png'.format(self._alg_name))
        plt.legend()
        plt.close()

    # def plot_models(self):
    #     plot_model(model=self.model, to_file=self._log_dir + '/model.png', show_shapes=True)


    def plot_series(self):
        preds = np.load(self._log_dir+'pd.npy')
        gt = np.load(self._log_dir+'gt.npy')
        if preds.shape[1] == 1 and gt.shape[1] == 1:
            pd.DataFrame(preds).to_csv(self._log_dir + "prediction_values.csv", header=['PM2.5'], index=False)
            pd.DataFrame(gt).to_csv(self._log_dir + "grouthtruth_values.csv", header=['PM2.5'], index=False)
        else:
            pd.DataFrame(preds).to_csv(self._log_dir + "prediction_values.csv", header=['PM10','PM2.5'], index=False)
            pd.DataFrame(gt).to_csv(self._log_dir + "grouthtruth_values.csv", header=['PM10','PM2.5'], index=False)

        for i in range(preds.shape[1]):
            plt.plot(preds[:, i], label='preds')
            plt.plot(gt[:, i], label='gt')
            plt.legend()
            plt.savefig(self._log_dir + '[result_predict]output_dim_{}.png'.format(str(i+1)))
            plt.close()