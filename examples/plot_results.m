data_kl_trpo = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_22_23_02_57_0001/progress.csv',1,0);
%/home/vgomez/rllab/data/local/experiment/experiment_2017_02_21_16_57_45_0001/progress.csv',1,0);
data_npireps = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_22_23_03_14_0001/progress.csv',1,0);
%/home/vgomez/rllab/data/local/experiment/experiment_2017_02_21_16_53_50_0001/progress.csv',1,0);
data_npireps2 = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_10_37_22_0001/progress.csv',1,0);

h = figure(1);
clf;
plot(data_npireps(:,1))
hold on;
plot(data_kl_trpo(:,6))
plot(data_npireps2(:,8))
%set(gca,'xscale','log');
title('Average Return')
xlabel('Iteration')

legend('nat PIREPS','KL-TRPO','nat PIREPS2', 'location','northwest')
grid on;

saveas(h,'fig1.eps','psc2')
