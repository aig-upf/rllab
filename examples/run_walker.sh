#!/bin/bash
source activate acrobotrllab
#source activate rllab4

keyword='Walkerepsilondelta'


N=100

n_parallel=7

stduncontrolled=1.
lambd=0.1
n_itr=100
algorithm='Algorithm2'


for epsilon in 0.01 0.1 0.001 
do
for seed in 11 22 33 44 55 66 77 88 99 1234
do

PoF='trpo_naive'
optim='cg'
delta=0.05
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &

PoF='J'
optim='cg'

delta=0.05
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &


delta=0.01
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &

delta=0.005
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0 >out1 2>out2 &

delta=0.1
python KL_anneal_walker.py $keyword $delta $epsilon $N $seed $n_parallel $algorithm $PoF $optim $stduncontrolled $lambd $n_itr 0

done
done