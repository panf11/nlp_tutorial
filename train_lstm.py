import os
# import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import time
# import torch.nn.functional as F
import torch.optim as optim
# from torch.autograd import Variable

from torchtext import data
from torchtext.data import Iterator, BucketIterator
from sklearn.metrics import roc_auc_score
# from transformers import BertTokenizer


class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            if self.y_vars is not None:  # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class BILSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


def roc_metrics(preds, y):
    preds = torch.sigmoid(preds)
    auc = 0
    for i in np.arange(6):
        auc += roc_auc_score(y.cpu().numpy()[:, i], preds.cpu().detach().numpy()[:, i])
    return auc/6.


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / correct.shape[0] / correct.shape[1]
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    preds = torch.Tensor().cuda()
    y_true = torch.Tensor().cuda()
    for x, y in iterator:
        optimizer.zero_grad()
        # predictions = model(batch.x).squeeze(1)
        predictions = model(x[0], x[1])
        loss = criterion(predictions, y)
        acc = binary_accuracy(predictions, y)
        preds = torch.cat((preds, predictions), dim=0)
        y_true = torch.cat((y_true, y), dim=0)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    auc = roc_metrics(preds, y_true)
    return epoch_loss / len(iterator), epoch_acc / len(iterator), auc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    preds = torch.Tensor().cuda()
    y_true = torch.Tensor().cuda()
    with torch.no_grad():
        for x, y in iterator:
            # predictions = model(batch.x).squeeze(1)
            predictions = model(x[0], x[1])
            loss = criterion(predictions, y)
            acc = binary_accuracy(predictions, y)
            preds = torch.cat((preds, predictions), dim=0)
            y_true = torch.cat((y_true, y), dim=0)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    auc = roc_metrics(preds, y_true)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), auc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    # tokenizer = lambda x: x.split()
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    TEXT = data.Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    trainval_datafields = [("id", None),  # we won't be needing the id, so we pass in None as the field
                           ("comment_text", TEXT), ("toxic", LABEL),
                           ("severe_toxic", LABEL), ("threat", LABEL),
                           ("obscene", LABEL), ("insult", LABEL),
                           ("identity_hate", LABEL)]
    test_datafields = [("id", None),  # we won't be needing the id, so we pass in None as the field
                       ("comment_text", TEXT)]
    SEED = 1
    BATCH_SIZE = 64

    data_dir = '/media/feng/storage/Downloads/jigsaw'
    train_data = data.TabularDataset(path=os.path.join(data_dir, 'train.csv'),
                                     format='csv', skip_header=True, fields=trainval_datafields)
    # valid_data = data.TabularDataset(path=os.path.join(data_dir, 'train.csv'),
    #                                  format='csv', skip_header=True, fields=tv_datafields)
    train_data, valid_data = train_data.split(split_ratio=0.8, stratified=False, strata_field='toxic',
                                              random_state=random.seed(SEED))

    test_data = data.TabularDataset(
        path=os.path.join(data_dir, "test.csv"),  # the file path
        format='csv',
        skip_header=True,
        # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
        fields=test_datafields)

    MAX_VOCAB_SIZE = 25_000
    print("Building vocabulary!!!!")
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    train_iter, val_iter = BucketIterator.splits((train_data, valid_data),
                                                 batch_sizes=(BATCH_SIZE, BATCH_SIZE),
                                                 device=torch.device("cuda"),
                                                 sort_key=lambda x: len(x.comment_text),
                                                 sort_within_batch=True,
                                                 repeat=False)
    test_iter = Iterator(test_data, batch_size=BATCH_SIZE, device=torch.device("cuda:0"), sort=False,
                         sort_within_batch=False,
                         repeat=False)

    train_dl = BatchWrapper(train_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult",
                                                         "identity_hate"])
    valid_dl = BatchWrapper(val_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult",
                                                       "identity_hate"])
    test_dl = BatchWrapper(test_iter, "comment_text", None)

    # em_sz = 100
    # nh = 500
    # # nl = 3
    # model = SimpleLSTMBaseline(nh, emb_dim=em_sz)

    # INPUT_DIM = len(TEXT.vocab)
    # EMBEDDING_DIM = 100
    # HIDDEN_DIM = 256
    # OUTPUT_DIM = 6
    #
    # model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 6
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = BILSTM(INPUT_DIM,
                    EMBEDDING_DIM,
                    HIDDEN_DIM,
                    OUTPUT_DIM,
                    N_LAYERS,
                    BIDIRECTIONAL,
                    DROPOUT,
                    PAD_IDX)
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    print(model.embedding.weight.data)

    model.cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    N_EPOCHS = 100

    best_valid_loss = float('inf')

    # for epoch in range(N_EPOCHS):
    #
    #     start_time = time.time()
    #
    #     train_loss, train_acc, train_auc_score = train(model, train_dl, optimizer, criterion)
    #     valid_loss, valid_acc, valid_auc_score = evaluate(model, valid_dl, criterion)
    #
    #     end_time = time.time()
    #
    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #
    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(model.state_dict(), 'tut1-model.pt')
    #
    #     print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    #     print(f'\tTrain Loss: {train_loss:.3f} | Train Auc: {train_auc_score * 100:.2f}%')
    #     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Auc: {valid_auc_score * 100:.2f}%')

    model.load_state_dict(torch.load('tut1-model.pt'))

    test_loss, test_acc, test_auc_score = evaluate(model, valid_dl, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Auc: {test_auc_score * 100:.2f}%')
