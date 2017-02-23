%data_kl_trpo = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_22_23_02_57_0001/progress.csv',1,0);
%/home/vgomez/rllab/data/local/experiment/experiment_2017_02_21_16_57_45_0001/progress.csv',1,0);
%data_npireps = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_22_23_03_14_0001/progress.csv',1,0);
%/home/vgomez/rllab/data/local/experiment/experiment_2017_02_21_16_53_50_0001/progress.csv',1,0);
%data_npireps2 = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_10_37_22_0001/progress.csv',1,0);
data_npireps = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_14_49_46_0001/progress.csv',1,0);
data_kl_trpo = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_14_37_48_0001/progress.csv',1,0);
data_kl_trpo = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_14_56_29_0001/progress.csv',1,0);


data_npireps_eta_0_05 = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_15_53_17_0001/progress.csv',1,0);
data_npireps_eta_0_2 = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_16_13_51_0001/progress.csv',1,0);

data_kl_trpo_total_cost = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_18_15_01_0001/progress.csv',1,0);
data_npireps_total_cost = csvread('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_18_26_08_0001/progress.csv',1,0);
data_npireps_total_cost_str = csvread(,0,0);
fileID = fopen('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_18_26_08_0001/progress.csv','r');
A = fscanf(fileID,'%s');
fclose(fileID);

h = figure(1);
clf;
plot(data_npireps(:,3))
hold on;
plot(data_kl_trpo(:,11))

%set(gca,'xscale','log');
title('Average Return')
xlabel('Iteration')

legend('nat PIREPS','KL-TRPO', 'location','northwest')
grid on;

saveas(h,'fig1.eps','psc2')

h = figure(2);
clf;
plot(data_npireps_eta_0_2(:,1))
hold on;
plot(data_npireps_eta_0_05(:,14))

%set(gca,'xscale','log');
title('Average Return')
xlabel('Iteration')

legend('nat PIREPS (\delta = 0.05)','nat PIREPS (\delta = 0.2)', 'location','northwest')
grid on;

saveas(h,'fig2.eps','psc2')

h = figure(3);
clf;
subplot(1,2,1);
plot(data_kl_trpo_total_cost(:,14));
ylabel('Average Reward');
hold on;
plot(data_npireps_total_cost(:,6));
grid on;
subplot(1,2,2);
plot(data_kl_trpo_total_cost(:,15))
ylabel('Total Cost');
hold on;
plot(data_npireps_total_cost(:,9));

grid on;
legend('KL-TRPO','Nat-PIREPS \delta=0.1')
saveas(h,'fig3.eps','psc2')


