import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import skipthoughts

class TransEncoder(nn.Module):

    def __init__(self, in_size, out_size, vocab):
        """Initialize encoder with structure parameters
        Args:
        """
        super(LSTMEncoder, self).__init__()
        self.linear = torch.nn.Linear(in_size, out_size)
        self.vocab = vocab
        self.stmodel = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(model)


    def __call__(self, s, xs, **kwargs):
        translated = self.translate(xs)
        vectors = self.encoder.encode(X)
        vectors = torch.from_numpy(vectors)
        return self.linear(vectors)
        


    def translate(self, xs):
        strings=[]

        # get reversed int to word mapping
        int2word = {}
        for k, v in self.vocab.items():
            int2word.setdefault(v, []).append(k); 

        for x in xs:
            s=""
            for item in x:
                s = s + int2word[int(item)] + " "

            strings.append(s[:-1])
        
        return strings