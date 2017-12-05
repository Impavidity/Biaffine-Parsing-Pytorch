import torch
from torchtext import data
import numpy as np
from pbase import app
from pbase import algorithm
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



arg_parser = Args()
args = arg_parser.get_args()

WORD = data.Field(batch_first=True)
PLEMMA = data.Field(batch_first=True)
PPOS = data.Field(batch_first=True)
DEP = data.Field(batch_first=True, use_vocab=False, preprocessing=lambda x: [int(y) for y in x], pad_token=-1)
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

    def __call__(self, output, batch):
        # return loss
        #return self.crit(output, batch.LABEL)
        size0 = output[0].size()[-1]
        arc_prob = output[0].view(-1, size0)
        print(output[1][:,1:,:,:].size())
        print(batch.DEP[:,1:].unsqueeze(2).unsqueeze(3).size())
        batch_size = output[1].size(0)
        temp = torch.index_select(output[1][0, 1:, :, :], 0, batch.DEP[0, 1:])
        print("TEMP", temp.size())
        label_prob = torch.stack( [torch.index_select(output[1][i,1:,:,:], 1, batch.DEP[i,1:]) for i in range(batch_size)])
        print(label_prob.size())
        return self.crit(arc_prob, batch.DEP.view(-1), ignore_index=-1) \
               + self.crit(label_prob, batch.LABEL[:,1:].view(-1))

def evaluator(name, pairs):
    #word = batch.WORD.cpu().data.numpy()[0]
    #prob = arc_prob.cpu().data.numpy()[0][:,:,0]
    #mask = np.not_equal(word, 1).astype(int)
    #sent_len= sum(mask)
    arc_right = 0
    label_right = 0
    total = 0
    if type(pairs) != list and type(pairs) == tuple:
        pairs = [pairs]
    for pair in pairs: # pair = (batch_output, batch_examples) batch_output = (arc_prob, rel_prob)
        for arc_prob, label_prob, example, depen, label in zip(pair[0][0].cpu().data.numpy(),
                                                 pair[0][1].cpu().data.numpy(),
                                                 pair[1].WORD.cpu().data.numpy(),
                                                 pair[1].DEP.cpu().data.numpy(),
                                                 pair[1].LABEL.cpu().data.numpy()):
            # output and example is one sentence
            print("ARC prob",arc_prob.shape)
            print("LABEL prob",label_prob.shape)
            print(example)
            print(depen)
            print(label)
            arc_prob = arc_prob[:,:,0]
            print("ARC prob", arc_prob.shape)
            word = example
            mask = np.not_equal(word, 1).astype(int)
            sent_len = sum(mask)
            true_arc = depen
            true_label = label
            arc_pred = algorithm.MaxSpanningTree(arc_prob, sent_len, mask)
            print(arc_pred)
            if name == 'train':
                arc_pred_for_label = true_arc
            else:
                arc_pred_for_label = arc_pred
            label_prob = label_prob[np.arange(len(arc_pred)), arc_pred]
            print("LABEL prob", label_prob.shape)
            label_pred = algorithm.rel_argmax(label_prob, sent_len, ROOT=LABEL.vocab.stoi["ROOT"])
            arc_comp = np.equal(arc_pred[1:sent_len], true_arc[1:sent_len])
            label_comp = np.equal(label_pred[1:sent_len], true_label[1:sent_len])
            total += sent_len -1
    return (arc_right/total,  label_right/total)





# The evaluator output is the input of metrics_comparison
# Used in parameters selection
def metrics_comparison(new_metrics, best_metrics):
    if best_metrics == None or new_metrics[1] > best_metrics[1]:
        return True
    return False


# The evaluator output is the input of log_printer
def log_printer(name,  metrics,epoch=None, iters=None):
    if name == 'train':
        print(name, "EPOCH : ", epoch, "ITER : ", iters,"UAS : ", metrics[0], "LAS : ", metrics[1])
    else:
        print(name, "UAS : ", metrics[0], "LAS : ", metrics[1])

trainer = Trainer(args=args, fields=fields, include_test=include_test)

trainer.prepare(model=Parser, optimizer=optimizer, criterion=criterion(),
                evaluator=evaluator, metrics_comparison=metrics_comparison, log_printer=log_printer)

trainer.train()




