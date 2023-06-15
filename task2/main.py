import argparse
from trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--device',type=int, default=0)
    parser.add_argument('--num_epochs',type=int,default=10)
    parser.add_argument('--eval_iter',type=int,default=1)
    parser.add_argument('--save_dir',type=str,default='../task3/interface_2/')
    return parser.parse_args()


def main(args):
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    args = get_args()
    main(args)