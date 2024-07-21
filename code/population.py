from search_space import HybridSearchSpace
from individual import Individual
import numpy as np
from gnn_model_manager import GNNModelManager
import random
import torch
from hyperopt import fmin, tpe, hp
from sklearn.metrics import precision_recall_curve, auc, roc_curve


class HyperparameterTuner:
    @staticmethod
    def tune_hyperparameters(logits1, logits2, logits3, logits4, logits5, logits6, logits7, space, data):
        def objective(params):
            x = sum(params[f'{i}'] * logits for i, logits in
                    enumerate([logits1, logits2, logits3, logits4, logits5, logits6, logits7], start=1))
            _, indices = torch.max(x.cuda(), dim=1)
            accuracy = (torch.sum(
                indices[data.val_mask.cuda()] == data.y.cuda()[data.val_mask.cuda()])).item() * 1.0 / 500
            return -accuracy

        best = fmin(objective, space, algo=tpe.suggest, max_evals=500)
        best_params = {f'{i}': best[f'{i}'] for i in range(1, 8)}
        x = sum(best_params[f'{i}'] * logits for i, logits in
                enumerate([logits1, logits2, logits3, logits4, logits5, logits6, logits7], start=1))
        y = torch.exp(x / sum(best_params.values()))
        _, test_indices = torch.max(y.cuda(), dim=1)
        test_accuracy = (torch.sum(
            test_indices[data.test_mask.cuda()] == data.y.cuda()[data.test_mask.cuda()])).item() * 1.0 / 1000
        return test_accuracy

    @staticmethod
    def tune_hyperparameters5(logits1, logits2, logits3, logits4, logits5, space, data):
        def objective(params):
            x = sum(params[f'{i}'] * logits for i, logits in
                    enumerate([logits1, logits2, logits3, logits4, logits5], start=1))
            _, indices = torch.max(x.cuda(), dim=1)
            val_acc = []
            val1 = (torch.sum(
                indices[data.val_mask.cuda()] == data.y.cuda()[data.val_mask.cuda()])).item() * 1.0 / 649
            val2 = (torch.sum(
                indices[data.val_mask2.cuda()] == data.y.cuda()[data.val_mask2.cuda()])).item() * 1.0 / 649
            val3 = (torch.sum(
                indices[data.val_mask3.cuda()] == data.y.cuda()[data.val_mask3.cuda()])).item() * 1.0 / 649
            val4 = (torch.sum(
                indices[data.val_mask4.cuda()] == data.y.cuda()[data.val_mask4.cuda()])).item() * 1.0 / 650
            val5 = (torch.sum(
                indices[data.val_mask5.cuda()] == data.y.cuda()[data.val_mask5.cuda()])).item() * 1.0 / 650
            val_acc.append(val1)
            val_acc.append(val2)
            val_acc.append(val3)
            val_acc.append(val4)
            val_acc.append(val5)
            accuracy = np.mean(val_acc)
            return -accuracy

        best = fmin(objective, space, algo=tpe.suggest, max_evals=500)
        best_params = {f'{i}': best[f'{i}'] for i in range(1, 6)}
        x = sum(best_params[f'{i}'] * logits for i, logits in
                enumerate([logits1, logits2, logits3, logits4, logits5], start=1))
        y = torch.exp(x / sum(best_params.values()))
        _, test_indices = torch.max(y.cuda(), dim=1)
        test_accuracy = (torch.sum(
            test_indices[data.val_mask.cpu() + data.train_mask.cpu()] == data.y.cuda()[
                data.val_mask.cpu() + data.train_mask.cpu()])).item() * 1.0 / 812
        y_true = data.y[data.val_mask.cpu() + data.train_mask.cpu()].cpu().detach()
        y_score = y[data.val_mask.cpu() + data.train_mask.cpu()][:, 1].cpu().detach()
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        fpr, tpr, th = roc_curve(y_true.cpu(), y_score.cpu(), pos_label=1)
        auprc = auc(recall, precision)
        auroc = auc(fpr, tpr)
        return test_accuracy, auprc, auroc


class Population(object):

    def __init__(self, args):

        self.args = args
        hybrid_search_space = HybridSearchSpace(self.args.num_gnn_layers)
        self.hybrid_search_space = hybrid_search_space

        # prepare data set for training the gnn model
        self.load_trining_data()
        for i in range(1, self.niche + 1):
            setattr(self, f'best_individuals{i}_net', [])
            setattr(self, f'best_individuals{i}_validate', [])
            setattr(self, f'best_individuals{i}_test', [])
            setattr(self, f'best_individuals{i}_all', [])

        # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'cora')
        # dataset = Planetoid(path, 'cora', transform=T.NormalizeFeatures())
        # data = dataset[0]
        # self.data = data

    def load_trining_data(self):
        if self.args.dataset in ['cora', 'citeseer', 'pubmed']:
            self.gnn_manager = GNNModelManager(self.args)
            self.data = self.gnn_manager.load_data(self.args.dataset)
            self.niche = 7
        if self.args.dataset in ['kegg', 'DawnNet', 'RegNetwork']:
            self.gnn_manager = GNNModelManager(self.args)
            self.data = self.gnn_manager.load_data_cancer(self.args.dataset)
            self.niche = 5

        # dataset statistics
        print(self.gnn_manager.data)

    def init_population(self):
        for j in range(1, self.niche + 1):
            globals()[f'struct_individuals{j}'] = []
            for i in range(self.args.niche_individuals):
                net_genes = self.hybrid_search_space.get_net_instance()
                net_genes[4] = 2 ** (j + 1)
                param_genes = self.hybrid_search_space.get_param_instance()
                #             param_genes = [self.args.lr, self.args.in_drop, self.args.weight_decay]
                instance = Individual(self.args, net_genes, param_genes)
                globals()[f'struct_individuals{j}'].append(instance)

    def cal_fitness(self):
        """calculate fitness scores of all individuals,
          e.g., the classification accuracy from GNN"""
        for i in range(1, self.niche + 1):
            for individual in globals()[f'struct_individuals{i}']:
                individual.cal_fitness(self.gnn_manager)

    def crossover_net(self):
        "produce offspring from parents for better net architecture"
        for i in range(1, self.niche + 1):
            globals()[f'best_individual{i}'] = globals()[f'struct_individuals{i}'][0]
            for elem_index, elem in enumerate(globals()[f'struct_individuals{i}']):
                if globals()[f'best_individual{i}'].get_fitness() < elem.get_fitness():
                    globals()[f'best_individual{i}'] = elem
        for j in range(1, self.niche + 1):
            for i in range(self.args.niche_individuals):
                y1 = random.randint(0, 1)
                z1 = random.randint(0, self.args.niche_individuals - 1)
                parent_gene_i = globals()[f'struct_individuals{j}'][i].get_net_genes()
                parent_gene_j = globals()[f'best_individual{j}'].get_net_genes()
                if y1 == 0:
                    parent_gene_j = globals()[f'struct_individuals{j}'][z1].get_net_genes()
                if parent_gene_i == globals()[f'best_individual{j}'].get_net_genes():
                    arr = [1, 2, 3, 4, 5, 6, 7]
                    if j in arr:
                        arr.remove(j)
                    else:
                        print('error')
                    n = random.choice(arr)
                    parent_gene_j = globals()[f'best_individual{n}'].get_net_genes()
                offspring_gene_i = parent_gene_i.copy()
                for k in range(10):
                    z2 = np.random.randint(0, 2)
                    if z2 == 0:
                        offspring_gene_i[k] = parent_gene_j[k]
                offspring_gene_i[4] = 2 ** (j + 1)
                offspring_i = Individual(self.args, offspring_gene_i,
                                         globals()[f'struct_individuals{j}'][i].get_param_genes())
                offspring_i.cal_fitness(self.gnn_manager)
                if offspring_i.get_fitness() > globals()[f'struct_individuals{j}'][i].get_fitness():
                    globals()[f'struct_individuals{j}'][i] = offspring_i
                else:
                    random_prob = np.random.uniform(0, 1, 1)
                    if random_prob <= self.args.mutate_prob:
                        index, gene = self.hybrid_search_space.get_one_net_gene()
                        offspring_i.mutation_net_gene(index, gene, 'struct')
                        offspring_i.cal_fitness(self.gnn_manager)
                        if offspring_i.get_fitness() > globals()[f'struct_individuals{j}'][i].get_fitness():
                            globals()[f'struct_individuals{j}'][i] = offspring_i

    def print_models(self, iter):

        print('===begin, current population ({} in {} generations)===='.format(
            (iter + 1), self.args.num_generations))
        for i in range(1, self.niche + 1):
            globals()[f'best_individual{i}'] = globals()[f'struct_individuals{i}'][0]
            for elem_index, elem in enumerate(globals()[f'struct_individuals{i}']):
                if globals()[f'best_individual{i}'].get_fitness() < elem.get_fitness():
                    globals()[f'best_individual{i}'] = elem
            getattr(self, f'best_individuals{i}_all').append(globals()[f'best_individual{i}'])
            getattr(self, f'best_individuals{i}_net').append(globals()[f'best_individual{i}'].get_net_genes())
            getattr(self, f'best_individuals{i}_validate').append(globals()[f'best_individual{i}'].get_fitness())
            getattr(self, f'best_individuals{i}_test').append(globals()[f'best_individual{i}'].get_test_acc())
        if self.niche == 7:
            best_individual = max(
                [best_individual1, best_individual2, best_individual3, best_individual4, best_individual5,
                 best_individual6, best_individual7],
                key=lambda individual: individual.get_fitness()
            )
        if self.niche == 5:
            best_individual = max(
                [best_individual1, best_individual2, best_individual3, best_individual4, best_individual5],
                key=lambda individual: individual.get_fitness()
            )
        cora_logits_sum = []
        for j in range(1, self.niche + 1):
            for i in range(0, self.args.niche_individuals):
                print('struct space: {}, validate_acc={}, test_acc={}, params={}'.format(
                    globals()[f'struct_individuals{j}'][i].get_net_genes(),
                    globals()[f'struct_individuals{j}'][i].get_fitness(),
                    globals()[f'struct_individuals{j}'][i].get_test_acc(),
                    globals()[f'struct_individuals{j}'][i].get_params()))
                globals()[f'cora_logits{j}'] = np.exp(globals()[f'best_individual{j}'].get_logits())
                cora_logits_sum.append(globals()[f'cora_logits{j}'])
        print(cora_logits_sum)
        x = 0
        for i in range(0, self.niche - 1):
            for j in range(i + 1, self.niche - 1):
                x += torch.mean(torch.abs(cora_logits_sum[i] - cora_logits_sum[j]))
        print(x)

        print('------the best model-------')
        print('struct space: {}, validate_acc={}, test_acc={}, params={}'.format(
            best_individual.get_net_genes(),
            best_individual.get_fitness(),
            best_individual.get_test_acc(),
            best_individual.get_params()))
        for i in range(1, self.niche + 1):
            print('struct space: {}, validate_acc={}, test_acc={}, params={}'.format(
                globals()[f'best_individual{i}'].get_net_genes(),
                globals()[f'best_individual{i}'].get_fitness(),
                globals()[f'best_individual{i}'].get_test_acc(),
                globals()[f'best_individual{i}'].get_params()))
            validate = getattr(self, f'best_individuals{i}_validate')
            test = getattr(self, f'best_individuals{i}_test')
            print(validate, test)

        print('====end====\n')

        return best_individual, x

    def evolve_net_cite(self):
        # initialize population
        self.init_population()
        # calculate fitness for population
        self.cal_fitness()

        actions = []
        train_accs = []
        test_accs = []
        od = []
        ensemble_accuracy = []

        for i in range(self.args.num_generations):
            # GNN structure evolution
            print('GNN structure evolution')
            self.crossover_net()  # crossover to produce offsprings
            best_individual, x = self.print_models(i)
            od.append(x)
            actions.append(best_individual.get_net_genes())
            train_accs.append(best_individual.get_fitness())
            test_accs.append(best_individual.get_test_acc())
            ensemble_logits1 = self.best_individuals1_all[i - 1].get_logits().detach()
            ensemble_logits2 = self.best_individuals2_all[i - 1].get_logits().detach()
            ensemble_logits3 = self.best_individuals3_all[i - 1].get_logits().detach()
            ensemble_logits4 = self.best_individuals4_all[i - 1].get_logits().detach()
            ensemble_logits5 = self.best_individuals5_all[i - 1].get_logits().detach()
            ensemble_logits6 = self.best_individuals6_all[i - 1].get_logits().detach()
            ensemble_logits7 = self.best_individuals7_all[i - 1].get_logits().detach()
            space = {
                f'{i}': hp.quniform(f'{i}', 0, 1, 0.01) for i in range(1, 8)
            }
            data = self.data
            test_accuracy = HyperparameterTuner.tune_hyperparameters(ensemble_logits1, ensemble_logits2,
                                                                     ensemble_logits3, ensemble_logits4,
                                                                     ensemble_logits5, ensemble_logits6,
                                                                     ensemble_logits7, space,
                                                                     data)
            print(test_accuracy)
            ensemble_accuracy.append(test_accuracy)
        print(od, ensemble_accuracy)
        print(actions)
        print(train_accs)
        print(test_accs)

    def evolve_net_cancer(self):
        # initialize population
        self.init_population()
        # calculate fitness for population
        self.cal_fitness()

        actions = []
        train_accs = []
        test_accs = []
        ensemble_accuracy = []
        ensemble_auprc = []
        ensemble_auroc = []

        for i in range(self.args.num_generations):
            # GNN structure evolution
            print('GNN structure evolution')
            self.crossover_net()  # crossover to produce offsprings
            best_individual, x = self.print_models(i)
            actions.append(best_individual.get_net_genes())
            train_accs.append(best_individual.get_fitness())
            test_accs.append(best_individual.get_test_acc())
            ensemble_logits1 = self.best_individuals1_all[i - 1].get_logits().detach()
            ensemble_logits2 = self.best_individuals2_all[i - 1].get_logits().detach()
            ensemble_logits3 = self.best_individuals3_all[i - 1].get_logits().detach()
            ensemble_logits4 = self.best_individuals4_all[i - 1].get_logits().detach()
            ensemble_logits5 = self.best_individuals5_all[i - 1].get_logits().detach()
            space = {
                f'{i}': hp.quniform(f'{i}', 0, 1, 0.01) for i in range(1, 6)
            }
            data = self.data
            test_accuracy, auprc, auroc = HyperparameterTuner.tune_hyperparameters5(ensemble_logits1, ensemble_logits2,
                                                                                    ensemble_logits3, ensemble_logits4,
                                                                                    ensemble_logits5, space,
                                                                                    data)
            print(test_accuracy, auprc, auroc)
            ensemble_accuracy.append(test_accuracy)
            ensemble_auprc.append(auprc)
            ensemble_auroc.append(auroc)

        print(actions)
        print(train_accs)
        print(test_accs)
        print(ensemble_accuracy)
        print(ensemble_auprc)
        print(ensemble_auroc)
