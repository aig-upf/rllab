#!/bin/bash
for seed in 10001 10123 11111 9876 88776634 77339922 99667236 1230748 92937 33428
do
for delta in 0.5 0.2 0.15 0.125 0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.0313 0.02 0.01 0.005 0.001 0.00000381
do
cat <<EOF >sbatch.temp
#!/bin/bash
#SBATCH -n 1 -c 1
#SBATCH -p snn
srun python npireps_double_pendulum_picked.py MiAvond_deltasweep $delta 0.001 2000 $seed 1 policy1 0 1 100
EOF
sbatch sbatch.temp
done
done


#!/bin/bash
for seed in 10001 10123 11111 9876 88776634 77339922 99667236 1230748 92937 33428
do
for delta in 0.1 0.08 0.0313 0.00000381
do
cat <<EOF >sbatch.temp
#!/bin/bash
#SBATCH -n 1 -c 1
#SBATCH -p snn
srun python npireps_double_pendulum_picked.py MiAvond_1000it $delta 0.001 2000 $seed 1 policy1 0 1 1000
EOF
sbatch sbatch.temp
done
done


#!/bin/bash
for seed in 10001 10123 11111 9876 88776634 77339922 99667236 1230748 92937 33428
do
for delta in 0.1 0.08 0.0313 0.00000381
do
cat <<EOF >sbatch.temp
#!/bin/bash
#SBATCH -n 1 -c 1
#SBATCH -p snn
srun python npireps_double_pendulum_picked.py MiAvond_1000it_largeNN $delta 0.001 2000 $seed 1 policy2 0 1 1000
EOF
sbatch sbatch.temp
done
done

#!/bin/bash
for seed in 10001 10123 11111 9876 88776634 77339922 99667236 1230748 92937 33428
do
for delta in 0.1 0.08 0.0313 0.00000381
do
cat <<EOF >sbatch.temp
#!/bin/bash
#SBATCH -n 1 -c 1
#SBATCH -p snn
srun python npireps_double_pendulum_time_picked.py MiAvond_1000it_withtime $delta 0.001 2000 $seed 1 policy1 0 1 1000
EOF
sbatch sbatch.temp
done
done


#!/bin/bash
for seed in 10001 10123 11111 9876 88776634 77339922 99667236 1230748 92937 33428
do
for delta in 0.1 0.08 0.0313 0.00000381
do
cat <<EOF >sbatch.temp
#!/bin/bash
#SBATCH -n 1 -c 1
#SBATCH -p snn
srun python npireps_double_pendulum_time_picked.py MiAvond_1000it_largeNN_withtime $delta 0.001 2000 $seed 1 policy2 0 1 1000
EOF
sbatch sbatch.temp
done
done