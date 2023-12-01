import argparse


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

class Arsg(object):

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--seed',
                                 type=int,
                                 default=100,
                                 help="random_seed")
        self.parser.add_argument('--batch_size',
                                 type=int,
                                 default=32,
                                 help="batch size")
        self.parser.add_argument('--lr',
                                 type=float,
                                 default=1e-4,
                                 help="learning rate")
        self.parser.add_argument('--weight_decay',
                                 type=float,
                                 default=1e-5,
                                 help="weight decay")
        self.parser.add_argument('--epochs',
                                 type=int,
                                 default=100,
                                 help="epochs")
        # parser.add_argument('--use_arg', type=boolean_string, default=False, help='whether use the args')
        self.parser.add_argument('--use_log',
                                 type=boolean_string,
                                 default=True,
                                 help='whether use the log')
        self.parser.add_argument('--load_model',
                                 type=boolean_string,
                                 default=False,
                                 help='whether read the model')
        self.parser.add_argument('--load_epoch',
                                 type=int,
                                 default=20,
                                 help='the epoch to be read')
        self.parser.add_argument('--augment',
                                 type=str,
                                 default='none',
                                 help='')

        self.parser.add_argument('--qkv_proj',
                                 type=str,
                                 default='linear',
                                 help='qkv projection type')
        self.parser.add_argument('--ffn_type',
                                 type=str,
                                 default='leff',
                                 help='feed forward network type')
        
    def parse(self, ):
        
        args = self.parser.parse_args()

        return args
