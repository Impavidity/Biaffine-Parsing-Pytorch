import torch
from torch import nn
from pbase import layer


class SRL(nn.Module):
    def __init__(self, config):
        super(SRL, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.word_num, config.word_dim)
        self.pre_embed = nn.Embedding(config.word_num, config.word_dim)
        self.pre_embed.weight.requires_grad = False
        self.pos_embed = nn.Embedding(config.pos_num, config.pos_dim)
        self.lem_embed = nn.Embedding(config.lem_num, config.lem_dim)
        self.srl_embed = nn.Embedding(config.srl_num, config.srl_dim)
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
        if config.cuda:
            self.srl_range = torch.autograd.Variable(torch.LongTensor(list(range(self.config.srl_num)))).cuda(config.gpu)
        else:
            self.srl_range = torch.autograd.Variable(torch.LongTensor(list(range(self.config.srl_num))))
        self.SRLParaMLP = layer.MLP(in_features=config.srl_dim+config.lem_dim,
                                    out_features=config.lstm_hidden * 4,
                                    activation=nn.LeakyReLU(0.1),
                                    dropout=0.33)


    def forward(self, x):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.embed_dropout(self.embed(x.WORD))
        x_pre_embed = self.embed_dropout(self.pre_embed(x.WORD))
        x_embed = x_word_embed + x_pre_embed
        x_pos_embed = self.embed_dropout(self.pos_embed(x.PPOS))
        x_lem_embed = self.embed_dropout(self.lem_embed(x.PLEMMA))
        x_is_verb_embed = self.is_verb_embed(x.INDICATOR)
        x_lexical = torch.cat((x_embed, x_pos_embed, x_lem_embed, x_is_verb_embed), dim=2)
        outputs, (ht, ct) = self.lstm(x_lexical)
        # output = (batch_size, sentence_length, hidden_size * num_direction)
        # Pass the is_verb == 1 through config
        predicate_index = (torch.eq(x.INDICATOR, self.config.ISVERB)).float()
        predicate_hidden = torch.sum(outputs * predicate_index.unsqueeze(dim=2), dim=1).unsqueeze(dim=1)
        # predicate_hidden = (batch_size, 1, hidden_size * num_direction)
        token_to_keep = (1 - torch.eq(x.WORD, self.config.PAD).float()).unsqueeze(dim=2)
        # token_to_keep = (batch_size, sentence_length, 1)
        predicate_broadcast = token_to_keep * predicate_hidden
        mlp_input = torch.cat([outputs, predicate_broadcast], dim=2)
        # mlp_input = (batch_size, sentence_length, lstm_hidden * 4)

        # Generate parameters for SRL
        batch_size = x.WORD.size(0)
        role_embed = self.srl_embed(self.srl_range)
        role_embed = torch.cat([role_embed.unsqueeze(dim=0)] * batch_size, dim=0)
        # role_embed = (srl_num, srl_dim)
        # print(predicate_index)
        # print(x.WORD)
        # print((predicate_index * x.WORD.float()).size())
        predicate_word_id = torch.sum(predicate_index * x.PLEMMA.float(), dim=1).long()
        predicate_lem = self.embed_dropout(self.lem_embed(predicate_word_id))
        predicate_lem = torch.cat([predicate_lem.unsqueeze(dim=1)] * self.config.srl_num, dim=1)
        pre_para = torch.cat([role_embed, predicate_lem], dim=2)
        # pre_para = (batch_size, srl_num, pre_para_dim)
        srl_para = self.SRLParaMLP(pre_para).transpose(1, 2)
        # srl_para = (batch_size, srl_num, lstm_hidden * 4)
        logit = torch.matmul(mlp_input, srl_para)
        # logit = (batch_size, sentence_length, srl_num)
        return logit






        



