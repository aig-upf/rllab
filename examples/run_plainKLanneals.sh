#!/bin/bash
source activate rllab4

which='Acrobot-v1'
keyword='debug'
#keyword='debug_introspection'
#keyword='squareerrorintrospection'
#keyword='KLgympendulum1'

epsilon=0.01
N=100
seed=10
n_parallel=20

stduncontrolled=1.0
lambd=1.
n_itr=100

#algorithm='trpo'
#PoF='trpo_naive'
optim='cg'
#PoF='KL_CE'

algorithm='Algorithm'
PoF='J'
delta=0.05
#python plainKL_anneal_gymCat.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 $which
python plainKL_anneal_acrobot.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0

#for seed in 554 372 204 12 9834 6 34 24 #18 10
#do

#algorithm='Algorithm'
#PoF='KL_CE'
#delta=0.2
#python plainKL_anneal_gymCat.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 $which

#done