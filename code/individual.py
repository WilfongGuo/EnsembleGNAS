import numpy as np
import torch


class Individual(object):

    def __init__(self, args, net_genes, param_genes):

        self.args = args
        self.net_genes = net_genes
        self.param_genes = param_genes

    def get_net_genes(self):
        return self.net_genes

    def get_param_genes(self):
        return self.param_genes

    def cal_fitness(self, gnn_manager):
        # run gnn to get the classification accuracy as fitness
        if self.args.dataset in ['cora', 'citeseer', 'pubmed']:
            val_acc, test_acc, val_loss, logits_best, params = gnn_manager.train(self.net_genes, self.param_genes)
        if self.args.dataset in ['kegg', 'DawnNet', 'RegNetwork']:
            val_acc, test_acc, val_loss, logits_best = gnn_manager.train1(self.net_genes, self.param_genes)
        self.fitness = val_acc
        self.test_acc = test_acc
        self.logits = torch.tensor(logits_best, dtype=torch.float64).cpu()
        self.params = torch.tensor(params)

    def get_fitness(self):
        return self.fitness

    def get_test_acc(self):
        return self.test_acc

    def get_logits(self):
        return self.logits

    def get_params(self):
        return self.params

    def mutation_net_gene(self, mutate_point, new_gene, type='struct'):
        if type == 'struct':
            self.net_genes[mutate_point] = new_gene
        elif type == 'param':
            self.param_genes[mutate_point] = new_gene
        else:
            raise Exception("wrong mutation type")
