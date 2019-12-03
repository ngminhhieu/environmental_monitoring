from model.deep_layers_lstm import lstm_enc, lstm_dec
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import plot_model

def lstm_layers_ed_model_construction(input_dim, output_dim, rnn_units, dropout, optimizer, log_dir, rnn_layers, is_training=True):
        # Model
        encoder_inputs = Input(shape=(None, input_dim))
        _, encoder_states = lstm_enc(encoder_inputs, rnn_unit=rnn_units,
                                                        rnn_depth=rnn_layers,
                                                        rnn_dropout=dropout)
        decoder_inputs = Input(shape=(None, output_dim))
        layers, decoder_outputs, _ = lstm_dec(decoder_inputs, rnn_unit=rnn_units,
                                                                     rnn_depth=rnn_layers,
                                                                     rnn_dropout=dropout,
                                                                     init_states=encoder_states)

        decoder_dense = Dense(output_dim, activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        if is_training:
            return model
        else:
            print("Load model from: {}".format(log_dir))
            model.load_weights(log_dir + 'best_model.hdf5')
            model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

            # Inference encoder_model
            encoder_model = Model(encoder_inputs, encoder_states)

            # Inference decoder_model
            decoder_states_inputs = []
            decoder_states = []
            decoder_outputs = decoder_inputs
            for i in range(rnn_layers):
                decoder_state_input_h = Input(shape=(rnn_units,))
                decoder_state_input_c = Input(shape=(rnn_units,))
                decoder_states_inputs += [decoder_state_input_h, decoder_state_input_c]
                d_o, state_h, state_c = layers[i](decoder_outputs, initial_state=decoder_states_inputs[2*i:2*(i+1)])
                decoder_outputs = d_o
                decoder_states += [state_h, state_c]
            decoder_outputs = decoder_dense(decoder_outputs)
            decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)

            plot_model(model=encoder_model, to_file=log_dir + '/encoder.png', show_shapes=True)
            plot_model(model=decoder_model, to_file=log_dir + '/decoder.png', show_shapes=True)

            return model, encoder_model, decoder_model