import torch
import torch.nn as nn
import torch.legacy.nn as luann
import sys

class MaxPool(nn.Module):
    def __init__(self, dim=1):
        super(MaxPool, self).__init__()
        self.dim = dim
    
    def forward(self, input):
        return torch.max(input, self.dim)[0]

    def __repr__(self):
        return self.__class__.__name__ +'('+ 'dim=' + str(self.dim) + ')'

class View(nn.Module):
    def __init__(self, *sizes):
        super(View, self).__init__()
        self.sizes_list = sizes

    def forward(self, input):
        return input.view(*self.sizes_list)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'sizes=' + str(self.sizes_list) + ')'

class Transpose(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input):
        return input.transpose(self.dim1, self.dim2).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'between=' + str(self.dim1) + ',' + str(self.dim2) + ')'

class CNNModel(nn.Module):
    def __init__(self, vocab_size, num_labels, emb_size, w_hid_size, h_hid_size, win, batch_size, with_proj=False):
        super(CNNModel, self).__init__()

        self.model = nn.Sequential()
        self.model.add_module('transpose', Transpose())
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.model.add_module('emb', self.embed)
        if with_proj:
            self.model.add_module('view1', View(-1, emb_size))
            self.model.add_module('linear1', nn.Linear(emb_size, w_hid_size))
            self.model.add_module('relu1', nn.ReLU())
        else:
            w_hid_size = emb_size

        self.model.add_module('trans2', Transpose(1, 2))

        conv_nn = nn.Conv1d(w_hid_size, h_hid_size, win, padding=1)
        self.model.add_module('conv', conv_nn)
        self.model.add_module('relu2', nn.ReLU())

        self.model.add_module('max', MaxPool(2))

        self.model.add_module('view4', View(batch_size, -1))
        self.model.add_module('linear2', nn.Linear(h_hid_size, num_labels))
        self.model.add_module('softmax', nn.LogSoftmax())


    def forward(self, x):

        output = self.model.forward(x)

        return output

