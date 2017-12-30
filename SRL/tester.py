from pbase import app
from main import Args, fields, include_test, params, evaluator, log_printer

arg_parser = Args()
args = arg_parser.get_args()
tester = app.TestAPP(args=args, fields=fields, include_test=include_test,
                     build_vocab_params=params)

tester.prepare(evaluator=evaluator, log_printer=log_printer)
print(tester.model)
tester.test()