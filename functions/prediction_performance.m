% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Yuxiao Yang and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the one-step-ahead Kalman prediction of neural activity based on
% the adaptively identified model parameters. It finally computes the
% Explained Variance (EV) as a measure for prediction performance based on Kalman prediction.
% Refer to Ahmadipour et al 2020, Section 2.4 or Yang et al 2020, section 4.3 for more details.

% We obtain the one-step-ahead prediction of y(t+1) (y_pred(t+1|t)) using a recursive Kalman predictor:
%                           x_pred(t+1|t)=A(t)x_pred_(t|t-1)+K(t)(y(t)-C(t-1)x_pred_(t|t-1));
%                           y_pred(t+1|t)=C(t)x_pred_(t+1|t);
%

%   Inputs:
%     - (1) y: neural observations (matrix with size ny by T)
%     - (2) sys_id: cell containing identified model parameters at all
%     time steps (size T_sysId)
%     - (3) x_initial: initial state of the Kalman predictor
%     - (4) t_start_eval: from this time step to the end we include the
%     prediction error in computing the performance measures.
%   Outputs:
%     - (1) EV: Explained Variance of all neural observations computed
%     based on Kalman prediction.
%     - (2) mean_EV: Average of EV over neural observations
%     - (3) RPE: Relative Prediction Error of all neural observations
%     computed based on Kalman prediction.
%     - (4) mean_RPE: Average of EV over neural observations.
%     - (5) x_pred_last: Final state of the Kalman predictor

function [EV, mean_EV, RPE, mean_RPE, x_pred_last] = prediction_performance(y, sys_id, x_initial, t_start_eval)

    T_sysId = size(sys_id, 1); % number of identified systems
    ny = size(y, 1); % number of neural observations
    pred_error = zeros(ny, T_sysId); % matrix of Kalman prediction error
    y_pred = zeros(ny, T_sysId);
    shiftIndex_y = size(y, 2) - T_sysId + 1;
    y = y(:, shiftIndex_y:end); % allign y with identified systems
    nx = size(sys_id{1, 1}.A, 1);

    if isempty(x_initial)
        x_initial = zeros(nx, 1);
    end

    x_pred_last = x_initial;

    for kk = 1:T_sysId
        A = sys_id{kk, 1}.A;
        C = sys_id{kk, 1}.C;
        K = sys_id{kk, 1}.K;

        if all(abs(eig(A - K * C)) < 1) % if identified parameters A,K, and C determine an stable system

            y_pred(:, kk) = C * x_pred_last;
            pred_error(:, kk) = y(:, kk) - y_pred(:, kk);
            x_pred_new = A * x_pred_last + K * pred_error(:, kk);
            x_pred_last = x_pred_new;
        else % if identified parameters A,K, and C do not determine an stable system
            y_pred(:, kk) = C * x_pred_last;
            pred_error(:, kk) = y(:, kk) - y_pred(:, kk);
            x_pred_new = A * x_pred_last; % forward prediction
            x_pred_last = x_pred_new;

        end

    end

    squared_error = pred_error(:, t_start_eval:end).^2;
    mse = mean(squared_error, 2);
    var_e = var(y(:, t_start_eval:end), 0, 2);
    EV = 1 - mse ./ var_e; % Compute Explained Variance (EV) based on Kalman prediction error
    RPE = sqrt(mse) ./ sqrt(var_e); % Compute Relative Prediction Error (RPE) based on Kalman prediction error
    mean_EV = mean(EV); % mean over neural observations
    mean_RPE = mean(RPE); % mean over neural observations
end
