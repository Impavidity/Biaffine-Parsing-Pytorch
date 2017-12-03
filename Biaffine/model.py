import torch
from torch import nn
from pbase import layer
import torch.nn.functional as F


class Parser(nn.Module):
    def __init__(self, config):
        super(Parser, self).__init__()
        self.embed = nn.Embedding(config.word_num, config.word_dim)
        self.pre_embed = nn.Embedding(config.word_num, config.word_dim)
        self.pre_embed.weight.requires_grad = False
        self.pos_embed = nn.Embedding(config.pos_num, config.pos_dim)
        # TODO: Set dropout for word embedding
        self.lstm = layer.LSTM(
            input_size=config.word_dim + config.pos_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.num_lstm_layer,
            dropout=config.lstm_dropout,
            bidirectional=True
        )
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
        # TODO: Set dropout for MLP
        self.arc_biaffine = layer.Biaffine(config.arc_mlp_size, config.arc_mlp_size, 1, bias=(True, False, False))
        self.rel_biaffine = layer.Biaffine(config.rel_mlp_size, config.rel_mlp_size, config.label_size,
                                           bias=(True, True, True))

    def forward(self, x, train=True):
        # x = (sequence length, batch size, dimension of embedding)
        x_word_embed = self.embed(x.WORD)
        x_pre_embed = self.pre_embed(x.WORD)
        x_pos_embed = self.pos_embed(x.POS)
        x_lexical = torch.cat((x_word_embed + x_pre_embed, x_pos_embed), dim=2)
        outputs, (ht, ct) = self.lstm(x_lexical)
        # output = (sentence_length, batch_size, hidden_size * num_direction)
        # ht = (layer * direction, batch, hidden_dim)
        # ct = (layer * direction, batch, hidden_dim)
        x_arc_dep = self.mlp_arc_dep(outputs)
        # x_dep = (sentence_length, batch_size, mlp_size)
        x_arc_head = self.mlp_arc_head(outputs)
        # x_head = (sentence_length, batch_size, mlp_size)
        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        x_rel_dep = self.mlp_rel_dep(outputs)
        x_rel_head = self.mlp_rel_head(outputs)
        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        return arc_logit, rel_logit_cond












