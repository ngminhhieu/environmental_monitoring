from keras.layers import Dense, Input, LSTM
from keras.models import Model
from keras.utils import plot_model

def lstm_ed_model_construction(input_dim, output_dim, rnn_units, dropout, optimizer, log_dir, is_training=True):
        # Model
        encoder_inputs = Input(shape=(None, input_dim), name='encoder_input')
        encoder = LSTM(rnn_units, return_state=True, dropout=dropout)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, output_dim), name='decoder_input')
        decoder_lstm = LSTM(rnn_units, return_sequences=True, return_state=True, dropout=dropout)
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        decoder_dense = Dense(output_dim, activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        if is_training:
            return model
        else:
            print("Load model from: {}".format(log_dir))
            model.load_weights(log_dir + 'best_model.hdf5')
            model.compile(optimizer=optimizer, loss='mse')

            # Inference encoder_model
            encoder_model = Model(encoder_inputs, encoder_states)

            # Inference decoder_model
            decoder_state_input_h = Input(shape=(rnn_units,))
            decoder_state_input_c = Input(shape=(rnn_units,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [state_h, state_c]
            decoder_outputs = decoder_dense(decoder_outputs)

            decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

            plot_model(model=encoder_model, to_file=log_dir + '/encoder.png', show_shapes=True)
            plot_model(model=decoder_model, to_file=log_dir + '/decoder.png', show_shapes=True)

            return model, encoder_model, decoder_model
