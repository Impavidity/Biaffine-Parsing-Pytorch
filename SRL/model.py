import torch
from torch import nn
from pbase import layer


class SRL:
    def __init__(self, config):
        super(SRL, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.word_num, config.word_dim)
        self.pre_embed = nn.Embedding(config.word_num, config.word_dim)
        self.pre_embed.weight.requires_grad = False
        self.pos_embed = nn.Embedding(config.pos_num, config.pos_dim)
        self.lem_embed = nn.Embedding(config.lem_num, config.lem_dim)
        self.is_verb_embed = nn.Embedding(config.is_verb_num, config.is_verb_dim)
        self.embed_dropout = nn.Dropout(p=0.33)
        self.lstm = layer.LSTM(
            input_size=config.word_dim + config.pos_dim + config.lem_dim + config.is_verb_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.num_lstm_layer,
            dropout=config.lstm_dropout,
            batch_first=True,
            bidirectional=True
        )


    def forward(self, x):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.embed_dropout(self.embed(x.WORD))
        x_pre_embed = self.embed_dropout(self.pre_embed(x.WORD))
        x_embed = x_word_embed + x_pre_embed
        x_pos_embed = self.embed_dropout(self.pos_embed(x.PPOS))
        x_lem_embed = self.embed_dropout(self.lem_embed(x.LEMMA))
        x_is_verb_embed = self.is_verb_embed(x.ISVERB)
        x_lexical = torch.cat((x_embed, x_pos_embed, x_lem_embed, x_is_verb_embed), dim=2)
        outputs, (ht, ct) = self.lstm(x_lexical)
        # output = (batch_size, sentence_length, hidden_size * num_direction)
        # Pass the is_verb == 1 through config
        predicate_index = torch.equal(x.ISVERB, self.config.ISVERB)
        predicate_hidden = torch.sum(outputs * predicate_index, dim=1)
        



