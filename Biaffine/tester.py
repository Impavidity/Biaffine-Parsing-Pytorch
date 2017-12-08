from pbase import app
from pbase import algorithm
import numpy as np
import os
from torchtext import data
from main import Args, evaluator, log_printer, fields, include_test, params


arg_parser = Args()
args = arg_parser.get_args()
tester = app.TestAPP(args=args, fields=fields, include_test=include_test, build_vocab_params=params)

WORD = fields[0][1]
LABEL = fields[4][1]
WORD_DICT = WORD.vocab.itos
LABEL_DICT = LABEL.vocab.itos

def output_parser(name, pairs):
    arc_right = 0
    label_right = 0
    total = 0
    fout = open(os.path.join(args.result_path, name), "w")
    for pair in pairs: # pair = (batch_output, batch_examples) batch_output = (arc_prob, rel_prob)
        for arc_prob, label_prob, example, depen, label in zip(pair[0][0].cpu().data.numpy(),
                                                 pair[0][1].cpu().data.numpy(),
                                                 pair[1].WORD.cpu().data.numpy(),
                                                 pair[1].DEP.cpu().data.numpy(),
                                                 pair[1].LABEL.cpu().data.numpy()):
            # output and example is one sentence
            arc_prob = arc_prob[:,:,0]
            word = example
            mask = np.not_equal(word, 1).astype(int)
            sent_len = sum(mask)
            true_arc = depen
            true_label = label
            arc_pred = algorithm.MaxSpanningTree(arc_prob, sent_len, mask)
            if name == 'train':
                arc_pred_for_label = true_arc
            else:
                arc_pred_for_label = arc_pred
            label_prob = label_prob[np.arange(len(arc_pred)), arc_pred]
            label_pred = algorithm.rel_argmax(label_prob, sent_len, ROOT=LABEL.vocab.stoi["ROOT"])
            for token, gdep, glabel, pdep, plabel in zip(word, depen, label, arc_pred, label_pred):
                if gdep == -1:
                    continue
                fout.write("_\t{}\t_\t_\t_\t_\t_\t_\t_\t{}\t_\t{}\n".format(WORD_DICT[token], pdep, LABEL_DICT[plabel]))
            fout.write("\n")
            arc_comp = np.equal(arc_pred[1:sent_len], true_arc[1:sent_len])
            label_comp = np.equal(label_pred[1:sent_len], true_label[1:sent_len]) * arc_comp
            arc_right += np.sum(arc_comp)
            label_right += np.sum(label_comp)
            total += sent_len -1
    return (arc_right/total,  label_right/total)

tester.prepare(evaluator=evaluator, log_printer=log_printer, output_parser=output_parser)
tester.test()
