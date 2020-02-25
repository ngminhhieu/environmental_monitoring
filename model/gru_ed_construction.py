from keras.layers import Dense, Input, GRU
from keras.models import Model
from keras.utils import plot_model

def gru_ed_model_construction(input_dim, output_dim, rnn_units, dropout, optimizer, log_dir, is_training=True):
        # Model
        encoder_inputs = Input(shape=(None, input_dim), name='encoder_input')
        encoder = GRU(rnn_units, return_state=True, dropout=dropout)
        encoder_outputs, state = encoder(encoder_inputs)

        encoder_states = [state]

        decoder_inputs = Input(shape=(None, output_dim), name='decoder_input')
        decoder_gru = GRU(rnn_units, return_sequences=True, return_state=True, dropout=dropout)
        decoder_outputs, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_states)

        decoder_dense = Dense(output_dim, activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        if is_training:
            return model
        else:
            model.load_weights(log_dir + 'best_model.hdf5')
            model.compile(optimizer=optimizer, loss='mse')

            # Inference encoder_model
            encoder_model = Model(encoder_inputs, encoder_states)

            # Inference decoder_model
            decoder_state_input = Input(shape=(rnn_units,))
            decoder_outputs, state = decoder_gru(decoder_inputs, initial_state=decoder_state_input)
            decoder_states = [state]
            decoder_outputs = decoder_dense(decoder_outputs)

            decoder_model = Model([decoder_inputs] + decoder_state_input, [decoder_outputs] + decoder_states)

            plot_model(model=encoder_model, to_file=log_dir + '/encoder.png', show_shapes=True)
            plot_model(model=decoder_model, to_file=log_dir + '/decoder.png', show_shapes=True)

            return model, encoder_model, decoder_model
