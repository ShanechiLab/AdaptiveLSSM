% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Yuxiao Yang and Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is the implementation of Adaptive LSSM (fitting) Algorithm
% in Ahmadipour et al 2020 and Yang et al 2020 (Algorithm 1).
% It adaptively identifies model parameters of a time-varying Linear State Space Model (LSSM)
% for all time steps of a trial. The identification at time step t is
% done based on observations of time steps 1 to t and recursively.

%   Inputs:
%     - (1) data: neural observations (matrix with size ny by trial length, where ny is the number of neural observations)
%     - (2) beta: forgetting factor (learning rate)
%     - (3) horizon: a parameter of the conventional Subspace System
%     Identification algorithm. Refer to Yang et al 2020 Appendix B for its determination.
%     - (4) nx: latent state dimension of LSSM
%   Outputs:
%     - (1) sys_id: a cell array of size T. Each cell contains the adaptively identified LSSM
%     parameters at a different time step of the trial, i.e.
%     A(t),C(t),NoiseCovariance(t), K(t) (Kalman Gain) with the following LSSM:
%                           x(t+1)=A(t)x(t)+w(t);
%                           y(t)=C(t)x(t)+v(t);

%                           with E{[w(t),v(t)][w(t)' v(t)']}
%                           =[Q(t),S(t);S(t)',R(t)]
%                           =NoiseCovariance(t)
%

function [sys_id] = AdaptiveLSSMFittingAlgorithm_wholeTrial(data, beta, horizon, nx, L_initial)

    T = size(data, 2) - (2 * horizon) + 1; % number of columns in the block Hankel matrix in the the conventional Subspace System Identification algorithms.
    ny = size(data, 1); % number of neural observations.
    sys_id = cell(T, 1);
    U1_past = rand(ny * horizon, nx);

    % Iterative identification of time-varying LSSM parameters for all time steps:
    L_past = L_initial;

    for t = 1:T
        % ****************************************************
        %               Algorithm 1 - STEP 2 - Yang et al 2020
        % ****************************************************
        % Updating the L matrix of LQ decomposition of Block Hankel matrix of data, when a new data (neural observation) comes in
        y_new = zeros(ny * horizon * 2, 1);

        for k1 = 1:2 * horizon
            y_new((k1 - 1) * ny + 1:k1 * ny, 1) = data(:, (t - 1) + k1);
        end

        [L_current] = lq_updating(L_past, y_new, beta);
        L_current = L_current(1:2 * horizon * ny, 1:2 * horizon * ny);

        % ***********************************************************
        %               Algorithm 1 - STEP 3 to 10 - Yang et al 2020
        % ***********************************************************
        % Identification of time-varying LSSM parameters  at time step t
        [sys_id{t, 1}, U1_current] = AdaptiveLSSMFittingAlgorithm_singleStep(L_current, ny, nx, horizon, beta, t, U1_past);
        U1_past = U1_current;
        L_past = L_current;
    end

end
