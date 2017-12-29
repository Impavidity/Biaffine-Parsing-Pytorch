import torch
from torchtext import data
import numpy as np
from pbase import app, logger
from model import SRL

WORD = data.Field(batch_first=True, lower=True)
PLEMMA = data.Field(batch_first=True)
PPOS = data.Field(batch_first=True)
DEP = data.Field(batch_first=True, use_vocab=False, preprocessing=lambda x: [int(y) for y in x], pad_token=-1)
LABEL = data.Field(batch_first=True)
INDICATOR = data.Field(batch_first=True)
SLABEL = data.Field(batch_first=True)
fields = [('WORD', WORD), ('PLEMMA', PLEMMA), ('PPOS', PPOS), ('DEP', DEP), ('LABEL', LABEL),
          ('INDICATOR', INDICATOR), ('SLABEL', SLABEL)]
include_test = [False, False, False, False, False, False, False]
params = [{"min_freq":2}, {}, {}, {}, {}, {}, {}]

class Args(app.ArgParser):
    def __init__(self):
        super(Args, self).__init__(description="SRL", patience=10000)
        self.parser.add_argument('--word_dim', type=int, default=100)
        self.parser.add_argument('--pos_dim', type=int, default=100)
        self.parser.add_argument('--lem_dim', type=int, default=100)
        self.parser.add_argument('--srl_dim', type=int, default=100)
        self.parser.add_argument('--is_verb_dim', type=int, default=10)
        self.parser.add_argument('--lstm_hidden', type=int, default=400)
        self.parser.add_argument('--num_lstm_layer', type=int, default=4)
        self.parser.add_argument('--lstm_dropout', type=float, default=0.33)
        self.parser.add_argument('--vector_cache', type=str,
                                 default="/mnt/collections/p8shi/dev/biaffine/Biaffine/data/glove.100d.conll09.pt")
        self.parser.add_argument('--lr', type=float, default='2e-3')
        self.parser.add_argument('--tensorboard', type=str, default='logs')

class Trainer(app.TrainAPP):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.config.word_num = len(self.WORD.vocab)
        self.config.pos_num = len(self.PPOS.vocab)
        self.config.lem_num = len(self.PLEMMA.vocab)
        self.config.is_verb_num = len(self.INDICATOR.vocab)
        self.config.srl_num = len(self.SLABEL.vocab)
        self.config.PAD = WORD.vocab.stoi['<pad>']
        self.config.ISVERB = INDICATOR.vocab.stoi['1']
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
        print(self.SLABEL.vocab.itos)
        print(self.config)

class optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.lr, betas=(0.9, 0.9), eps=1e-12)
        l = lambda epoch: 0.75 ** (epoch // 8)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def zero_grad(self):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()

    def schedule(self):
        self.scheduler.step()
        print("learning rate : ", self.scheduler.get_lr(), self.scheduler.base_lrs)


class criterion:
    def __init__(self):
        self.crit = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def __call__(self, output, batch):
        return self.crit(output.view(-1, output.size(2)), batch.SLABEL.view(-1, 1)[:,0])

def evaluator(name, pairs):
    # pair = (batch_output, batch_example)
    if type(pairs) != list and type(pairs) == tuple:
        pairs = [pairs]
    n_correct = 0
    n_total = 0
    n_predicted = 0
    acc_total = 0
    acc_right = 0
    for (output, batch) in pairs:
        acc_right += (torch.max(output, 2)[1].view(batch.SLABEL.size()).data == batch.SLABEL.data).sum(dim=1).sum()
        size_list = list(batch.SLABEL.size())
        acc_total += size_list[0] * size_list[1]
        pred = torch.max(output, 2)[1].view(batch.SLABEL.size()).cpu().data.numpy()
        gold = batch.SLABEL.cpu().data.numpy()
        for pred_sent, gold_sent in zip(pred, gold):
            for pred_token, gold_token in zip(pred_sent, gold_sent):
                if gold_token == SLABEL.vocab.stoi['<pad>']:
                    continue
                if pred_token == gold_token and gold_token != SLABEL.vocab.stoi['_']:
                    n_correct += 1
                if pred_token != SLABEL.vocab.stoi['_']:
                    n_predicted += 1
                if gold_token != SLABEL.vocab.stoi['_']:
                    n_total += 1
    if n_predicted == 0:
        precision = 0
    else:
        precision = n_correct / n_predicted
    if n_total == 0:
        recall = 0
    else:
        recall = n_correct / n_total
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    if acc_total == 0:
        acc = 0
    else:
        acc = acc_right / acc_total
    return acc, precision, recall, f1



def metrics_comparison(new_metrics, best_metrics):
    if best_metrics == None or new_metrics[3] > best_metrics[3]:
        return True
    return False

def log_printer(name, metrics, loss, epoch=None, iters=None):
    if name == 'train':
        print("{}\tEPOCH : {}\tITER : {}\tACC : {}\tP : {}\tR : {}\tF : {}\tNearest batch training LOSS : {}".format(
            name, epoch, iters, metrics[0], metrics[1], metrics[2], metrics[3], loss
        ))
        step = int(iters.split('/')[0]) + int(iters.split('/')[1]) * (epoch-1)
        log.scalar_summary(tag="loss", value=loss, step=step)
    else:
        if loss == None:
            print("{}\tACC : {}\tP : {}\tR : {}\tF : {}".format(name, metrics[0], metrics[1], metrics[2], metrics[3]))
        else:
            print("{}\tACC : {}\tP : {}\tR : {}\tF : {}".format(name, metrics[0], metrics[1], metrics[2], metrics[3], loss))
        if iters != None and epoch != None and loss != None:
            step = int(iters.split('/')[0]) + int(iters.split('/')[1]) * (epoch - 1)
            log.scalar_summary(tag="valid_loss", value=loss, step=step)

if __name__=='__main__':
    arg_parser = Args()
    args = arg_parser.get_args()
    log = logger.Logger(args.tensorboard)
    crit = criterion()
    trainer = Trainer(args=args, fields=fields, include_test=include_test,
                      build_vocab_params=params)
    trainer.prepare(model=SRL, optimizer=optimizer, criterion=crit,
                    evaluator=evaluator, metrics_comparison=metrics_comparison,
                    log_printer=log_printer)
    trainer.train()
