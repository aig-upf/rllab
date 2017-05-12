#!/bin/bash
source activate rllab4

keyword='KLanneallongerruns'

epsilon=0.1
N=200
seed=1708
n_parallel=1

stduncontrolled=1.0
lambd=0.1
n_itr=100

delta=0.3

algorithm='Algorithm2'
PoF='KL_CE'
optim='firstorder'

nohup python KL_anneal.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &



algorithm='Algorithm1'
PoF='KL_CE'
optim='firstorder'

nohup python KL_anneal.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &



algorithm='Algorithm2'
PoF='KL_CE2'
optim='firstorder'

nohup python KL_anneal.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &


delta=0.5

algorithm='Algorithm2'
PoF='KL_CE'
optim='firstorder'

nohup python KL_anneal.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &



algorithm='Algorithm1'
PoF='KL_CE'
optim='firstorder'

nohup python KL_anneal.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &



algorithm='Algorithm2'
PoF='KL_CE2'
optim='firstorder'

nohup python KL_anneal.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &


delta=0.01

algorithm='Algorithm2'
PoF='KL_CE'
optim='firstorder'

nohup python KL_anneal.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &



algorithm='Algorithm1'
PoF='KL_CE'
optim='firstorder'

nohup python KL_anneal.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &



algorithm='Algorithm2'
PoF='KL_CE2'
optim='firstorder'

nohup python KL_anneal.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &


