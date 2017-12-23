import torch
from torch import nn
from torch.autograd import Variable
from pbase import layer
import numpy as np
from collections import defaultdict

class Relation(nn.Module):
    def __init__(self, config):
        super(Relation, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.word_num, config.word_dim)
        self.pre_embed = nn.Embedding(config.word_num, config.word_dim)
        self.pos_embed = nn.Embedding(config.pos_num, config.pos_dim)
        self.char_embed = layer.CharEmbedding(config.char_num, config.char_dim)
        # After lookup, it should be 4D
        self.entity_embed = nn.Embedding(config.entity_num, config.entity_dim)
        # Use Word + Pre method here
        self.char_compose = layer.CharCNN(
            num_of_conv=1,
            in_channels=1,
            out_channels=100,
            kernel_size=2,
            in_features=config.char_dim,
            out_features=config.compose_char_dim,
            active_func=nn.Tanh()
        )
        self.lexical_lstm = layer.LSTM(
            input_size= config.word_dim + config.pos_dim + config.compose_char_dim,
            hidden_size= config.lexical_lstm_hidden,
            num_layers= config.lexical_num_lstm_layer,
            dropout= config.lstm_dropout,
            batch_first= True,
            bidirectional= True
        )
        self.entity_lstm = layer.LSTM(
            input_size= config.entity_dim,
            hidden_size= config.entity_lstm_hidden,
            num_layers= config.entity_num_lstm_layer,
            dropout= config.lstm_dropout,
            batch_first= True,
            bidirectional= False
        )
        self.entity_mlp = layer.MLP(
            in_features= config.lexical_lstm_hidden * 4 + config.entity_lstm_hidden * 2,
            out_features= config.table_hidden,
            activation= nn.LeakyReLU(0.1),
            dropout=0.3,
        )
        self.entity_final_layer = nn.Sequential(
            nn.Linear(config.table_hidden, config.final_hidden),
            nn.BatchNorm1d(config.final_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(config.dropout),
            nn.Linear(config.final_hidden, config.entity_num),
            nn.LogSoftmax()
        )

    def gen_pad(self, batch_size):
        if self.config.cuda:
            pad = Variable(torch.zeros(batch_size,
                                       self.config.lexical_lstm_hidden * 2 + self.config.entity_lstm_hidden * 2)).cuda()

        else:
            pad = Variable(torch.zeros(batch_size,
                                       self.config.lexical_lstm_hidden * 2 + self.config.entity_lstm_hidden * 2))

        return pad

    def gen_entity_hidden(self, batch_size):
        if self.config.cuda:
            entity_hidden = (Variable(torch.zeros(batch_size,
                                                  self.config.entity_num_lstm_layer,
                                                  self.entity_lstm_hidden)).cuda(),
                             Variable(torch.zeros(batch_size,
                                                  self.config.entity_num_lstm_layer,
                                                  self.entity_lstm_hidden)).cuda())
        else:
            entity_hidden = (Variable(torch.zeros(batch_size,
                                                  self.config.entity_num_lstm_layer,
                                                  self.entity_lstm_hidden)),
                             Variable(torch.zeros(batch_size,
                                                  self.config.entity_num_lstm_layer,
                                                  self.entity_lstm_hidden)))
        return entity_hidden

    def get_prev_entity_repr(self, batch_size, current_token_idx, lexical_lstm_output, entity_lstm_output, select_index):
        if current_token_idx == 0:
            pad = self.gen_pad(batch_size)
            return pad
        else:
            for sent_idx in range(batch_size):
                ner_tag = self.config.entity_itos[select_index[sent_idx][current_token_idx-1]][0]
                if ner_tag == 'o' or ner_tag == 's':
                    seg = torch.cat([entity_lstm_output[current_token_idx-1][sent_idx],
                                     lexical_lstm_output[sent_idx][current_token_idx-1]], dim=1)
                    return seg
                else: # b, m, e -> find the head
                    head = current_token_idx - 1
                    while head >= 0:
                        if self.config.entity_itos[select_index[sent_idx][head]][0] == 'b':
                            break
                        head -= 1
                    if head == -1:
                        print("Error in previous decoding : ")
                        for pre_index in range(current_token_idx):
                            print(self.config.entity_itos[select_index[sent_idx][pre_index]], ' ')
                        exit()
                    elif head == 0:
                        seg = torch.cat([entity_lstm_output[current_token_idx-1][sent_idx],
                                         lexical_lstm_output[sent_idx][current_token_idx-1]], dim=1)
                        return seg
                    else:
                        seg = torch.cat([entity_lstm_output[current_token_idx-1][sent_idx]
                                         - entity_lstm_output[head-1][sent_idx],
                                         lexical_lstm_output[sent_idx][current_token_idx-1]
                                         - lexical_lstm_output[sent_idx][head-1]], dim=1)
                        return seg





    def ner_decode(self, batch_size, each_sent_len, x_i, select_index, current_token_idx):
        valid_list = []
        for sent_idx in range(batch_size):
            valid = []
            if current_token_idx == 0:
                valid = [1] * len(self.config.ner_itos)
            else:
                prev_tag = self.config.ner_itos[select_index[sent_idx][current_token_idx-1]]
                allow_prefix = []
                # Do not consider type here, we give the decoding process the chance to change the type
                # Use majority vote in the evaluation
                if prev_tag[0] == 'o':
                    allow_prefix = ['o', 'b', 's']
                if prev_tag[0] == 's':
                    allow_prefix = ['o', 'b', 's']
                if prev_tag[0] == 'b':
                    allow_prefix = ['m', 'e']
                if prev_tag[0] == 'm':
                    allow_prefix = ['m', 'e']
                if prev_tag[0] == 'l':
                    allow_prefix = ['o', 's', 'b']
                if each_sent_len[sent_idx] <= current_token_idx:
                    allow_prefix = ['<']
                if each_sent_len[sent_idx]-1 == current_token_idx:
                    if 'm' in allow_prefix:
                        allow_prefix.remove('m')
                    if 'b' in allow_prefix:
                        allow_prefix.remove('b')
                # Will look ahead later.
                # TODO: if it is the last word in the sentence, then, it is impossible to be b or m
                for tag_id in range(len(self.config.ner_itos)):
                    if self.config.ner_itos[tag_id][0] in allow_prefix:
                        valid.append(1)
                    else:
                        valid.append(0)
            valid_list.append(valid)
        mask = torch.FloatTensor(valid_list)
        x_i = mask * x_i
        select_new = torch.max(x_i, 1)[1]
        for sent_idx in range(batch_size):
            select_index[sent_idx][current_token_idx] = select_new[sent_idx]
        return select_index


    def forward(self, x):
        word_embed = self.word_embed(x.WORD)
        pre_word_embed = self.pre_embed(x.WORD)
        pos_embed = self.pos_embed(x.POS)
        char_embed = self.char_embed(x.CHAR)
        char_compose_embed = self.char_compose(char_embed)
        lexical = torch.cat([word_embed + pre_word_embed, pos_embed, char_compose_embed], dim=2)
        outputs, (ht, ct) = self.lexical_lstm(lexical)
        # outputs = (batch_size, sentence_length, hidden_size * num_directions)
        batch_size = outputs.size(0)
        sent_length = outputs.size(1)
        each_sent_len = np.sum((np.not_equal(x.WORD.data.cpu().numpy(), self.config.pad_index)), axis=1)
        entity_lstm_output = []
        # Try schedule sampling here
        entity_hidden = self.gen_entity_hidden(batch_size)
        select_index = defaultdict(lambda : defaultdict(int))
        entity_logit_list = []
        for i in range(sent_length):
            prev_entity_repr = self.get_prev_entity_repr(batch_size, i, outputs, entity_lstm_output, select_index)
            concat = torch.cat([outputs[:,i,:], prev_entity_repr], dim=1)
            # (batch, hidden_size * num_directions) (batch_size, entity_lstm_hidden)
            # (batch_size, hidden_size * 2 + entity_lstm_hidden)
            x_i = self.entity_final_layer(self.entity_mlp(concat))
            entity_logit_list.append(x_i)
            # x_i = (batch_size, entity_num)
            # Create Allow Actions for decoding
            select_index = self.ner_decode(batch_size, each_sent_len, x_i, select_index, i)
            # select_index = (batch_size, 1)
            entity_i = self.entity_embed(select_index)
            # entity_i = (batch_size, 1, entity_dim)
            entity_output, entity_hidden = self.entity_lstm(entity_i, entity_hidden)
            entity_lstm_output.append(entity_output.sequeeze(1))
            # list of (batch_size, entity_dim)
        entity_logit = torch.stack(entity_logit_list, dim=1)
        return entity_logit












