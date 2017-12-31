from pbase import app
import os
import numpy as np
from main import Args, fields, include_test, params, evaluator, log_printer

arg_parser = Args()
args = arg_parser.get_args()
tester = app.TestAPP(args=args, fields=fields, include_test=include_test,
                     build_vocab_params=params)

# WORD = fields[0][1]
# WORD_DICT = np.array(WORD.vocab.itos)
#
# def output_parser(name, pairs):
#     fout = open(os.path.join(args.result_path, name), 'w')
#     for pair in pairs: # pair = (batch_output, batch_example)
#         for word in pair[1].WORD.cpu().data.numpy():
#             fout.write(" ".join(list(WORD_DICT[word])))
#             fout.write('\n')
#     fout.close()


tester.prepare(evaluator=evaluator, log_printer=log_printer)
print(tester.model)
tester.test()