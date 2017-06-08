#!/bin/bash
source activate rllab4

keyword='introspectionJvsCE'
#keyword='debug_introspection'
#keyword='squareerrorintrospection'
#keyword='KLgympendulum1'

epsilon=0.01
N=100
seed=88776634
n_parallel=5

stduncontrolled=1.0
lambd=0.1
n_itr=300


algorithm='Algorithm2'
PoF='J'
optim='cg'
delta=0.05
#python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 

python intro_KLanneal_gymCartpole.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 
