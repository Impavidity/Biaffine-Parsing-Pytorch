import torch
from torchtext import data
import numpy as np
from pbase import app
from model import Parser

class Args(app.ArgParser):
    def __init__(self):
        super(Args, self).__init__(description="Biaffine Parsing")
        self.parser.add_argument('--word_dim', type=int, default=100)
        self.parser.add_argument('--pos_dim', type=int, default=100)
        self.parser.add_argument('--lstm_hidden', type=int, default=400)
        self.parser.add_argument('--num_lstm_layer', type=int, default=3)
        self.parser.add_argument('--lstm_dropout', type=float, default=0.33)
        self.parser.add_argument('--arc_mlp_size', type=int, default=500)
        self.parser.add_argument('--rel_mlp_size', type=int, default=100)
        self.parser.add_argument('--vector_cache', type=str, default='data/glove.100d.conll09.pt')
        self.parser.add_argument('--lr', type=float, default='2e-3')

    def get_args(self):
        self.args = self.parser.parse_args(args=[])
        return self.args


arg_parser = Args()
args = arg_parser.get_args()

WORD = data.Field(batch_first=True)
PLEMMA = data.Field(batch_first=True)
PPOS = data.Field(batch_first=True)
DEP = data.Field(batch_first=True, use_vocab=False)
LABEL = data.Field(batch_first=True)
fields = [('WORD', WORD), ('PLEMMA', PLEMMA), ('PPOS', PPOS), ('DEP', DEP), ('LABEL', LABEL)]
include_test = [False, False, False, False, False]

class Trainer(app.TrainAPP):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.config.word_num = len(self.WORD.vocab)
        self.config.pos_num = len(self.PPOS.vocab)
        self.config.label_size = len(self.LABEL.vocab)
        stoi, vectors, dim = torch.load(self.config.vector_cache)
        match_embedding = 0
        self.WORD.vocab.vectors = torch.Tensor(len(self.WORD.vocab), dim)
        for i, token in enumerate(self.WORD.vocab.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                self.WORD.vocab.vectors[i] = vectors[wv_index]
                match_embedding += 1
            else:
                self.WORD.vocab.vectors[i] = torch.FloatTensor(self.config.word_dim).uniform_(-0.05, 0.05)
        print("Matching {} out of {}".format(match_embedding, len(self.WORD.vocab)))

    def prepare(self, **kwargs):
        super(Trainer, self).prepare(**kwargs)
        self.model.pre_embed.weight.data.copy_(self.WORD.vocab.vectors)
        print(self.model)
        print(self.LABEL.vocab.itos)


class optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr = config.lr)
        #self.optim = torch.optim.RMSprop(parameter, lr=1e-3)

    def zero_grad(self):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()

class criterion:
    # You need to do any modification to loss here
    # TODO: Might need to pass model parameters
    def __init__(self):
        self.crit = torch.nn.CrossEntropyLoss()

    def __call__(self, output, label):
        # return loss
        #return self.crit(output, batch.LABEL)
        return self.crit(output, label)

def evaluator(name):
    pass

# The evaluator output is the input of metrics_comparison
# Used in parameters selection
def metrics_comparison(new_metrics, best_metrics):
    if best_metrics == None or new_metrics[1] > best_metrics[1]:
        return True
    return False


# The evaluator output is the input of log_printer
def log_printer(name,  metrics,epoch=None, iters=None):
    if name == 'train':
        print(name,epoch,iters, metrics[0])
    else:
        print(name, metrics)

trainer = Trainer(args=args, fields=fields, include_test=include_test)

trainer.prepare(model=Parser, optimizer=optimizer, criterion=criterion(),
                evaluator=evaluator, metrics_comparison=metrics_comparison, log_printer=log_printer)




