import torch
from torch import nn
from torch.autograd import Variable
from pbase import layer
import numpy as np



class Parser(nn.Module):
    def __init__(self, config):
        super(Parser, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.word_num, config.word_dim)
        self.pre_embed = nn.Embedding(config.word_num, config.word_dim)
        self.pre_embed.weight.requires_grad = False
        self.pos_embed = nn.Embedding(config.pos_num, config.pos_dim)
        self.embed_dropout = nn.Dropout(p=0.33)
        self.lstm = layer.LSTM(
            input_size=config.word_dim + config.pos_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.num_lstm_layer,
            dropout=config.lstm_dropout,
            batch_first=True,
            bidirectional=True
        )
        self.batch_norm =  nn.BatchNorm1d(config.lstm_hidden * 2)
        self.mlp_arc_dep = layer.MLP(
            in_features = 2*config.lstm_hidden,
            out_features = config.arc_mlp_size,
            activation = nn.LeakyReLU(0.1))
        self.mlp_arc_head = layer.MLP(
            in_features = 2*config.lstm_hidden,
            out_features = config.arc_mlp_size,
            activation = nn.LeakyReLU(0.1))
        self.mlp_rel_dep = layer.MLP(
            in_features = 2*config.lstm_hidden,
            out_features = config.rel_mlp_size,
            activation = nn.LeakyReLU(0.1))
        self.mlp_rel_head = layer.MLP(
            in_features = 2*config.lstm_hidden,
            out_features = config.rel_mlp_size,
            activation = nn.LeakyReLU(0.1))
        self.mlp_dropout = nn.Dropout(p=0.33)
        # TODO: Set dropout for MLP
        self.arc_biaffine = layer.Biaffine(config.arc_mlp_size, config.arc_mlp_size, 1, bias=(True, False, False))
        self.rel_biaffine = layer.Biaffine(config.rel_mlp_size, config.rel_mlp_size, config.label_size,
                                           bias=(True, True, True))

    def embed_mask_generator(self, batch_size, seq_length):
        batch_word_dropout = []
        batch_tag_dropout = []
        for i in range(seq_length):
            word_mask = np.random.binomial(1, 1.0 - 0.33, batch_size).astype(np.float32)
            tag_mask = np.random.binomial(1, 1.0 - 0.33, batch_size).astype(np.float32)
            scale = 3.0 / (2.0 * word_mask + tag_mask + 1e-12)
            word_mask *= scale
            tag_mask *= scale
            word_mask = torch.from_numpy(word_mask)
            tag_mask = torch.from_numpy(tag_mask)
            batch_word_dropout.append(word_mask)
            batch_tag_dropout.append(tag_mask)
        batch_word_dropout = Variable(torch.stack(batch_word_dropout, dim=1).unsqueeze(dim=2), requires_grad=False)
        batch_tag_dropout = Variable(torch.stack(batch_tag_dropout, dim=1).unsqueeze(dim=2), requires_grad=False)
        if self.cuda != -1:
            return batch_word_dropout.cuda(self.config.gpu), batch_tag_dropout.cuda(self.config.gpu)
        else:
            return batch_word_dropout, batch_tag_dropout

    def forward(self, x):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.embed_dropout(self.embed(x.WORD))
        x_pre_embed = self.embed_dropout(self.pre_embed(x.WORD))
        x_embed = x_word_embed + x_pre_embed
        x_pos_embed = self.embed_dropout(self.pos_embed(x.PPOS))
        # if self.training:
        #     word_mask, tag_mask = self.embed_mask_generator(x_word_embed.size(0), x_word_embed.size(1))
        #     x_embed = x_embed * word_mask
        #     x_pos_embed = x_pos_embed * tag_mask
        x_lexical = torch.cat((x_embed, x_pos_embed), dim=2)
        outputs, (ht, ct) = self.lstm(x_lexical)
        size = outputs.size()
        outputs = self.batch_norm(outputs.contiguous().view(-1, size[2])).contiguous().view(size)
        # output = ( batch_size, sentence_length, hidden_size * num_direction)
        # ht = (batch, layer * direction, hidden_dim)
        # ct = (batch, layer * direction, hidden_dim)
        x_arc_dep = self.mlp_dropout(self.mlp_arc_dep(outputs))
        # x_dep = (batch_size, sentence_length, mlp_size)
        x_arc_head = self.mlp_dropout(self.mlp_arc_head(outputs))
        # x_head = (batch_size, sentence_length, mlp_size)
        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        x_rel_dep = self.mlp_dropout(self.mlp_rel_dep(outputs))
        x_rel_head = self.mlp_dropout(self.mlp_rel_head(outputs))
        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        return arc_logit, rel_logit_cond