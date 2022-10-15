import torch

from walker2d import *
import sys

run_walker = {
    'node': node_rnn_walker.main,
    'anode': anode_rnn_walker.main,
    'sonode': sonode_rnn_walker.main,
    'hbnode': hbnode_rnn_walker.main,
    'ghbnode': ghbnode_rnn_walker.main,
    'nesterovnode': nesterovnode_rnn_walker.main,
    'gnesterovnode': gnesterovnode_rnn_walker.main
}

all_models = {
    'walker': run_walker,
}


def main(ds='pv', model='hbnode', gpu=torch.device("cuda:0")):
    gpu = torch.device(f"cuda:{gpu}")
    all_models[ds][model](gpu)


if __name__ == '__main__':
    args = sys.argv[1:]
    assert len(args) >= 2, "Input format: python3 run.py task model gpu(optional for walker2d task)"
    print("Working on dataset {} using {} model and gpu {}".format(*args))
    main(*args)
