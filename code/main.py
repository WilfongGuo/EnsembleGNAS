import argparse
import time
import torch
import configs
import tensor_utils as utils
from population import Population


def main(args):
    torch.manual_seed(args.random_seed)
    utils.makedirs(args.dataset)
    pop = Population(args)
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        pop.evolve_net_cite()
    if args.dataset in ['kegg', 'DawnNet', 'RegNetwork']:
        pop.evolve_net_cancer()


if __name__ == "__main__":
    args = configs.build_args('GeneticGNN')
    main(args)
