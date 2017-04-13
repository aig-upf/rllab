#!/bin/bash
for seed in 10001 10123 11111 9876 88776634 77339922 99667236 1230748 92937 33428
do
for delta in 0.1 0.0313 0.00000381
do
cat <<EOF >sbatch.temp
#!/bin/bash
#SBATCH -n 1 -c 1
#SBATCH -p snn
source activate rllab4
srun python npireps_double_pendulum_picked.py shortsweep_normalcg_lambda_0.1_long $delta 0.001 2000 $seed 1 policy1 0 0.1 200
EOF
sbatch sbatch.temp
done
done