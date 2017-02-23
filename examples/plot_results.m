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
npireps_0_1_total_cost = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_18_26_08_0001/progress.csv','Total cost');
npireps_0_1_avg_return = get_data('/home/vgomez/rllab/data/local/experiment/experiment_2017_02_23_18_26_08_0001/progress.csv','AverageReturn');

h = figure(1);

clf;

subplot(1,2,1);
plot(kl_trpo_total_cost);
hold on;
plot(npireps_0_3_total_cost);
plot(npireps_0_1_total_cost);
grid on;
ylabel('Total Cost');

subplot(1,2,2);
plot(kl_trpo_avg_return)
ylabel('Average Reward');
hold on;
plot(npireps_0_3_avg_return);
plot(npireps_0_1_avg_return);

grid on;
legend('KL-TRPO','Nat-PIREPS \delta=0.3','location','southeast')
saveas(h,'fig3.eps','psc2')


end