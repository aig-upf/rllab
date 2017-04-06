#!/bin/bash
for seed in 10001 10123 11111 9876 88776634 77339922 99667236 1230748 92937 33428
do
for delta in 0.5 0.25 0.125 0.0625 0.0313 0.0156 0.0078 0.0039 0.0020 0.000977 0.000488 0.000244 0.000122 0.000061 0.000031 0.00001526 0.00000763 0.00000381
do
nohup python npirepsex2_double_pendulum_picked.py npireps newsweep1 $delta 0.001 10000 $seed 1 >out1 2>out2 &
done
done
    