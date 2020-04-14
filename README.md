# Install packages first
pip install -r requirements.txt

# To train each config
python main.py --config_file='config/sample.yaml' --mode=seq2seq_train

# To test
python main.py --config_file='config/sample.yaml' --mode=seq2seq_test

# To run model encoder decoder + genetic algorithm
python main.py --mode=ga_seq2seq