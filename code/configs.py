# -*- coding: utf-8 -*-

import argparse

def build_args(model_name):
    
    parser = argparse.ArgumentParser(description=model_name)
    register_default_args(parser)
    args = parser.parse_args()

    return args

def register_default_args(parser): 
    
    # general settings
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument("--dataset", type=str, default="cora", required=False,
                        help="The input dataset.")
    # settings for the genetic algorithm
    parser.add_argument('--niche_individuals', type=int, default=7,
                        help='the niche size')
    parser.add_argument('--num_generations', type=int, default=3,
                        help='number of evolving generations')
    parser.add_argument('--mutate_prob', type=float, default=0.02,
                        help='mutation probability')
    # settings for the gnn model
    parser.add_argument('--num_gnn_layers', type=int, default=2,
                        help='number of the GNN layers')
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")    
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=2,
                        help="number of training epochs")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--cuda", type=bool, default=True,
                        help="cuda")
    
    
#     parser.add_argument('--save_epoch', type=int, default=2)
#     parser.add_argument('--max_save_num', type=int, default=5)
#     parser.add_argument("--residual", action="store_false",
#                         help="use residual connection")
    
    
    
               