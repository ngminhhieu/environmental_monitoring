from keras.layers import LSTM


def lstm_enc(input, rnn_unit, rnn_depth, rnn_dropout):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """

    x = input
    states = []
    for i in range(rnn_depth):
        lstm_layer = LSTM(rnn_unit, return_sequences=True,
                                   return_state=True, name='LSTM_enc_{}'.format(i+1))
        x_rnn, state_h, state_c = lstm_layer(x)
        states += [state_h, state_c]        
        x = x_rnn
    return x, states


def lstm_dec(input, rnn_unit, rnn_depth, rnn_dropout, init_states):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    layers_lstm = []
    x = input    
    states = []
    for i in range(rnn_depth):
        lstm_layer = LSTM(rnn_unit, return_sequences=True,
                                   return_state=True, name='LSTM_dec_{}'.format(i+1))
        layers_lstm.append(lstm_layer)                                   
        x_rnn, state_h, state_c = lstm_layer(x, initial_state=init_states[2*i:2*(i+1)])
        states += [state_h, state_c]        
        x = x_rnn
    return layers_lstm, x, states
