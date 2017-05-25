#!/bin/bash
source activate rllab4

keyword='sweepAntagonist'
#keyword='debug_introspection'
#keyword='squareerrorintrospection'
#keyword='KLgympendulum1'

epsilon=0.01
N=100
seed=1708
n_parallel=30

stduncontrolled=1.0
lambd=0.1
n_itr=40



algorithm='Algorithm2_antagon'
PoF='J_antagon'
optim='cg'

delta=0.2
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0
delta=0.1
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0
delta=0.05
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0
delta=0.01
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0



algorithm='Algorithm2'
PoF='J'
optim='cg'

delta=0.2
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0
delta=0.1
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0
delta=0.05
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0
delta=0.01
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0



algorithm='Algorithm2'
PoF='trpo_naive'
optim='cg'
delta=0.2
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0



#keyword='KLwalker1'
keyword='KLgympendulum1'
keyword='debug_introspection'
epsilon=0.01
N=200
seed=1708
n_parallel=30

stduncontrolled=1.0
lambd=0.1
n_itr=30

delta=0.3
algorithm='Algorithm2'
PoF='KL_sym'
optim='firstorder'
#python KL_anneal_gympendulum.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0


