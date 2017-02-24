function plot_results

    function res = get_data(filename,key)
        fileID = fopen(filename,'r');
        C= textscan(fileID,'%s',17,'Delimiter',',');
        fclose(fileID);
        for i=1:numel(C{1})
            if strcmp(C{1}(i),key)
                idx = i;
                break
            end
        end
        data = csvread(filename,1,0);
        res = data(:,idx);
    end

kl_trpo_total_cost = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_18_15_01_0001/progress.csv','Total cost');
kl_trpo_avg_return = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_18_15_01_0001/progress.csv','AverageReturn');
npireps_0_3_total_cost = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_18_54_45_0001/progress.csv','Total cost');
npireps_0_3_avg_return = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_18_54_45_0001/progress.csv','AverageReturn');
npireps_0_1_total_cost = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_20_06_25_0001/progress.csv','Total cost');
%experiment_2017_02_23_18_26_08_0001
npireps_0_1_avg_return = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_20_06_25_0001/progress.csv','AverageReturn');
npireps_0_2_total_cost = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_20_18_52_0001/progress.csv','Total cost');
npireps_0_2_avg_return = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_20_18_52_0001/progress.csv','AverageReturn');
npireps_0_4_total_cost = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_19_50_55_0001/progress.csv','Total cost');
npireps_0_4_avg_return = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_19_50_55_0001/progress.csv','AverageReturn');
npireps_0_05_total_cost_eps_0_05 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_19_55_44_0001/progress.csv','Total cost');
npireps_0_05_avg_return_eps_0_05  = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_19_55_44_0001/progress.csv','AverageReturn');
npireps_0_05_total_cost_eps_0_01 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_20_16_29_0001/progress.csv','Total cost');
npireps_0_05_avg_return_eps_0_01  = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_20_16_29_0001/progress.csv','AverageReturn');

npireps_0_3_total_cost_2N = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_21_05_16_0001/progress.csv','Total cost');
npireps_0_3_avg_return_2N = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_21_05_16_0001/progress.csv','AverageReturn');
npireps_0_2_total_cost_2N = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_20_36_35_0001/progress.csv','Total cost');
npireps_0_2_avg_return_2N = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_20_36_35_0001/progress.csv','AverageReturn');
npireps_0_25_total_cost_2N = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_21_05_18_0001/progress.csv','Total cost');
npireps_0_25_avg_return_2N = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_21_05_18_0001/progress.csv','AverageReturn');
npireps_0_1_total_cost_2N = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_20_37_22_0001/progress.csv','Total cost');
npireps_0_1_avg_return_2N = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_20_37_22_0001/progress.csv','AverageReturn');

kl_trpo_total_cost_unc05 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_11_58_0001/progress.csv','Total cost');
kl_trpo_avg_return_unc05 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_11_58_0001/progress.csv','AverageReturn');
npireps_0_05_total_cost_unc05 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_17_38_0001/progress.csv','Total cost');
npireps_0_05_avg_return_unc05 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_17_38_0001/progress.csv','AverageReturn');
npireps_0_1_total_cost_unc05 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_12_51_0001/progress.csv','Total cost');
npireps_0_1_avg_return_unc05 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_12_51_0001/progress.csv','AverageReturn');
npireps_0_15_total_cost_unc05 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_21_21_0001/progress.csv','Total cost');
npireps_0_15_avg_return_unc05 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_21_21_0001/progress.csv','AverageReturn');

kl_trpo_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_52_50_0001/progress.csv','Total cost');
kl_trpo_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_52_50_0001/progress.csv','AverageReturn');

npireps_0_05_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_36_05_0001/progress.csv','Total cost');
npireps_0_05_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_36_05_0001/progress.csv','AverageReturn');

npireps_0_1_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_43_40_0001/progress.csv','Total cost');
npireps_0_1_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_43_40_0001/progress.csv','AverageReturn');

npireps_0_2_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_36_48_0001/progress.csv','Total cost');
npireps_0_2_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_36_48_0001/progress.csv','AverageReturn');

npireps_0_3_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_44_14_0001/progress.csv','Total cost');
npireps_0_3_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_22_44_14_0001/progress.csv','AverageReturn');

kl_trpo_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_12_37_15_0001/progress.csv','Total cost');
kl_trpo_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_12_37_15_0001/progress.csv','AverageReturn');
npireps_0_01_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_12_37_47_0001/progress.csv','Total cost');
npireps_0_01_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_12_37_47_0001/progress.csv','AverageReturn');
npireps_0_02_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_12_49_42_0001/progress.csv','Total cost');
npireps_0_02_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_12_49_42_0001/progress.csv','AverageReturn');
npireps_0_05_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_12_49_50_0001/progress.csv','Total cost');
npireps_0_05_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_12_49_50_0001/progress.csv','AverageReturn');

kl_trpo_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_16_45_01_0001/progress.csv','Total cost');
kl_trpo_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_16_45_01_0001/progress.csv','AverageReturn');
kl_trpo_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_16_58_54_0001/progress.csv','Total cost');
kl_trpo_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_16_58_54_0001/progress.csv','AverageReturn');
npireps_0_01_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_16_59_20_0001/progress.csv','Total cost');
npireps_0_01_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_16_59_20_0001/progress.csv','AverageReturn');
% npireps_0_02_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_15_03_48_0001/progress.csv','Total cost');
% npireps_0_02_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_15_03_48_0001/progress.csv','AverageReturn');
% npireps_0_05_total_cost_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_15_05_46_0001/progress.csv','Total cost');
% npireps_0_05_avg_return_unc05_lambda2 = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_24_15_05_46_0001/progress.csv','AverageReturn');

%2017_02_24_14_53
%2017_02_24_14_53_20

h = figure(2);

clf;

subplot(1,2,1);
plot(kl_trpo_total_cost_unc05_lambda2);
hold on;
plot(npireps_0_01_total_cost_unc05_lambda2);
% plot(npireps_0_02_total_cost_unc05_lambda2);
% plot(npireps_0_05_total_cost_unc05_lambda2);
grid on;
ylabel('Total Cost');
title('N = 400')

subplot(1,2,2);
plot(kl_trpo_avg_return_unc05_lambda2);
hold on;
plot(npireps_0_01_avg_return_unc05_lambda2);
% plot(npireps_0_02_avg_return_unc05_lambda2);
% plot(npireps_0_05_avg_return_unc05_lambda2);
ylabel('Average Reward');

grid on;
legend('KL-TRPO','Nat-PIREPS \delta=0.01','Nat-PIREPS \delta=0.02','Nat-PIREPS \delta=0.05','location','southeast')
saveas(h,'fig1.eps','psc2')

% 
% 
% h = figure(2);
% 
% clf;
% 
% subplot(1,2,1);
% plot(kl_trpo_total_cost_unc05);
% hold on;
% plot(npireps_0_05_total_cost_unc05);
% plot(npireps_0_1_total_cost_unc05);
% plot(npireps_0_15_total_cost_unc05);
% grid on;
% ylabel('Total Cost');
% title('N = 400')
% 
% subplot(1,2,2);
% plot(kl_trpo_avg_return_unc05)
% ylabel('Average Reward');
% hold on;
% plot(npireps_0_05_avg_return_unc05);
% plot(npireps_0_1_avg_return_unc05);
% plot(npireps_0_15_avg_return_unc05);
% 
% grid on;
% legend('KL-TRPO','Nat-PIREPS \delta=0.05','Nat-PIREPS \delta=0.1','Nat-PIREPS \delta=0.15','Nat-PIREPS \delta=0.3','location','southeast')
% saveas(h,'fig2.eps','psc2')
% 
% 
% 
% h = figure(3);
% 
% clf;
% 
% subplot(1,2,1);
% plot(kl_trpo_total_cost);
% hold on;
% plot(npireps_0_1_total_cost_2N);
% plot(npireps_0_2_total_cost_2N);
% plot(npireps_0_25_total_cost_2N);
% plot(npireps_0_3_total_cost_2N);
% grid on;
% ylabel('Total Cost');
% title('N = 400')
% 
% subplot(1,2,2);
% plot(kl_trpo_avg_return)
% ylabel('Average Reward');
% hold on;
% plot(npireps_0_1_avg_return_2N);
% plot(npireps_0_2_avg_return_2N);
% plot(npireps_0_25_avg_return_2N);
% plot(npireps_0_3_avg_return_2N);
% 
% grid on;
% legend('KL-TRPO','Nat-PIREPS \delta=0.1','Nat-PIREPS \delta=0.2','Nat-PIREPS \delta=0.25','Nat-PIREPS \delta=0.3','location','southeast')
% saveas(h,'fig3.eps','psc2')
% 
% 
% 
% h = figure(4);
% 
% clf;
% 
% subplot(1,2,1);
% plot(kl_trpo_total_cost);
% hold on;
% plot(npireps_0_4_total_cost);
% plot(npireps_0_3_total_cost);
% plot(npireps_0_1_total_cost);
% plot(npireps_0_2_total_cost);
% plot(npireps_0_05_total_cost_eps_0_05);
% plot(npireps_0_05_total_cost_eps_0_01);
% grid on;
% ylabel('Total Cost');
% title('N = 200')
% 
% subplot(1,2,2);
% plot(kl_trpo_avg_return)
% ylabel('Average Reward');
% hold on;
% plot(npireps_0_4_avg_return);
% plot(npireps_0_3_avg_return);
% plot(npireps_0_1_avg_return);
% plot(npireps_0_2_avg_return);
% plot(npireps_0_05_avg_return_eps_0_05);
% plot(npireps_0_05_avg_return_eps_0_01);
% 
% grid on;
% legend('KL-TRPO','Nat-PIREPS \delta=0.4','Nat-PIREPS \delta=0.3','Nat-PIREPS \delta=0.2','Nat-PIREPS \delta=0.1','Nat-PIREPS \delta=0.05, \epsilon=0.05','Nat-PIREPS \delta=0.05, \epsilon=0.01','location','southeast')
% saveas(h,'fig4.eps','psc2')
% 
% 

end