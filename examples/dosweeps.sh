#!/bin/bash
for delta in 0.1 0.01 0.001 0.0001 0.00001
do
python npirepsex_double_pendulum_picked.py npireps sweep0 $delta 0.001 2000 1 >out1 2>out2 &
done