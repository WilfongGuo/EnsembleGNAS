# Instruction of Ensemble-GNAS package

This package includes Python scripts and several citation networks datasets.

The input (case: cora) include: (1)Graph: cora. (2)niche_individuals:  the niche size of multi-modal architecture search algorithm. (3)num_generations: the maximum generation of  evolutionary search. (4)epochs: the maximum epochs of architecture evaluation.

The output results: (1)od: Output diversity of candidate networks found by search.

(2)ensemble-acccuracy: Classification accuracy of ensemble model.

(3)logits：Classification confidence scores for nodes from candidate networks and ensemble model.

Suggestions:

(1)Hardware suggestions for running this package: GPU Memory 16G or above.

(2)When users analyzed running this package, please note that:
Parameter setting of niche_individuals, num_generations, and epochs will affect the running time. With default parameters, Ensemble-GNAS takes about 40 minutes to search candidate networks and build the ensemble model. Users can decrease running time by modifying above parameter.

%    $Id: Main.m Created at 2021-11-15$ 
%   $Copyright (c) 2021 by School of Electrical Engineering, Zhengzhou University, Zhengzhou 450001, China$; 
%    $If any problem, please contact guowf@zzu.edu.cn for help. $

