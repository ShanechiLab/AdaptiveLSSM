% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Yuxiao Yang and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This script runs the Adaptive LSSM (fitting) algorithm in Ahmadipour et al 2020  and  Yang et al 2020
% for an example simulated non-stationary LSSM (brain network activity).
% Change  timeVaryingLSSM function arguments and beta_grid to explore the
% effect of beta on different non-stationary LSSMs.
% You can run the Non-adaptive LSSM algorithm by simply setting beta=1.
%%
clear all
nx = 3; % latent state dimension (order) of LSSM.
beta_grid = [0.96:0.005:0.99, 0.991:0.001:1]; % beta values to evaluate
trial_n = 4; % number of trials of neural activity to be evaluated from the same non-stationary LSSM (brain network activity(
%% Generating time-varying parameters of an LSSM

speed_nonStationarity = 1/5000; % speed of non-stationarity
T = 5000; % length of simulated neural activity
amp_range_nonStationarity = 0.1; % proportional to the amount of non-stationarity
angle_range_nonStationarity = pi; % proportional to the amount of non-stationarity
% Refer to the following function's description for more details about the
% above parameters
rng(3)
[sys_true] = timeVaryingLSSM(nx, T, speed_nonStationarity, amp_range_nonStationarity, angle_range_nonStationarity); % Refer to the function description for details about its arguments
ny = size(sys_true{1, 1}.C, 1); % Number of neural observations
%% Doing system identification and prediction performance evaluation
for trial_index = 1:trial_n
    data = generate_data(sys_true, []); % different trials from the same time-varying LSSM

    parfor beta_index = 1:length(beta_grid)
        horizon = ceil(nx / ny) + 1; % Yang et al 2020, Appendix B
        L_initial = tril(randn(2 * horizon * ny, 2 * horizon * ny)); %Generating a random lower triangular matrix as the initial L in the LQ decomposition
        % Running Adaptive LSSM algorithm at each of the time steps t of the whole trial
        [sys_id] = AdaptiveLSSMFittingAlgorithm_wholeTrial(data, beta_grid(beta_index), horizon, nx, L_initial);
        % Computing performance of the algorithm based on the adaptively identified model parameters "sys_id" at all time steps t
        [EV(trial_index, :, beta_index), mean_EV(trial_index, beta_index)] = prediction_performance(data, sys_id, [], 1);
    end

end

%% plotting Explained Variance (EV) averaged over trials as a function of the forgetting factor (beta, learning rate)
figure
hold on
errorbar(beta_grid, mean(mean_EV, 1), std(mean_EV, [], 1) ./ sqrt(trial_n));
[optimal_EV, optimal_beta_index] = max(mean(mean_EV, 1));
optimal_beta = beta_grid(optimal_beta_index);
h = plot(optimal_beta, optimal_EV, '*');
xlabel('Forgetting factor ($\beta$)', 'interpreter', 'latex');
ylabel('Prediction performance (EV)', 'interpreter', 'latex');
title(sprintf('latent state dimension of the fitted LSSM is %d', nx));
legend(h, 'optimal EV', 'interpreter', 'latex');
