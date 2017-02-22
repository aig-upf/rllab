data = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_21_16_57_45_0001/progress.csv',1,0);
plot(data(:,10))
set(gca,'xscale','log');
