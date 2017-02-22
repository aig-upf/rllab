data_trpo = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_21_16_57_45_0001/progress.csv',1,0);
data_npireps = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_21_16_53_50_0001/progress.csv',1,0);

h = figure(1);
clf;
plot(data_npireps(:,3))
hold on;
plot(data_trpo(:,13))
set(gca,'xscale','log');
title('Average Return')
xlabel('Iteration')

legend('npireps','trpo','location','northwest')

saveas(h,'fig1.eps','psc2')
