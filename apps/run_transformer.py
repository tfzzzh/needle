import needle as ndl
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

device = ndl.cpu_numpy()
corpus = ndl.data.Corpus("../data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=32, device=device, dtype="float32")
model = LanguageModel(20, len(corpus.dictionary), hidden_size=32, num_layers=1, seq_model='transformer', seq_len=20, device=device)
train_ptb(model, train_data, seq_len=20, n_epochs=10, device=device, lr=0.003, optimizer=ndl.optim.Adam)
evaluate_ptb(model, train_data, seq_len=20, device=device)
