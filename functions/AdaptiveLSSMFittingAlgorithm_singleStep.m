% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour,Yuxiao Yang, Maryam Shanechi
% Shanechi Lab, University of Southern California
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function implements Adaptive LSSM (fitting) Algorithm
% in Yang et al 2020 (Algorithm 1) and Ahmadipour et al 2020.
% Having only the L matrix in LQ decomposition at time step t, it estimates the time-varying LSSM parameters
% at this time-step.
%   Inputs:
%     - (1) L_mtx: L_mtx matrix in LQ decomposition
%     - (2) ny: dimension of neural observations in LSSM
%     - (3) nx: dimension of latent state in LSSM
%     - (4) horizon: a parameter of the algorithm. It determines the number of block rows of the block Hankel matrix in
%     conventional Subspace System Identification algorithms.
%     - (5) beta: forgetting factor in Adaptive LSSM fitting Algorithm
%     - (6) t: current time step
%     - (7) U1_prev: first nx columns of U matrix in SVD of the Ob matrix from the previous
%     time step t-1.
%   Outputs:
%     - (1) sys: an structure with the identified LSSM parameters A(t),C(t), NoiseCovariance(t) and K(t) (Kalman Gain) as its
%     fields,with the following Linear State Space Model (LSSM):
%                           x(t+1)=A(t)x(t)+w(t);
%                           y(t)=C(t)x(t)+v(t);
%                           with E{[w(t),v(t)][w(t)' v(t)']}
%                           =[Q(t),S(t);S(t)',R(t)]
%                           =NoiseCovariance(t)
%     - (2) U1_current: first nx columns of U matrix in SVD of the Ob in this time
%     step.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [sys, U1_current] = AdaptiveLSSMFittingAlgorithm_singleStep(L_mtx, ny, nx, horizon, beta, t, U1_prev);

    L_mtx = L_mtx(1:2 * horizon * ny, 1:2 * horizon * ny);

    % **************************************
    %               Algorithm 1 - STEP 3 and 4 - Yang et al 2020
    % **************************************

    % Compute the orthogonal projection Ob=Lf/Lp

    Lf = L_mtx(ny * horizon + 1:2 * ny * horizon, :);
    Lp = L_mtx(1:ny * horizon, :);
    Ob = Lf * Lp' * pinv(Lp * Lp') * Lp;

    % Compute the orthogonal projection Ob_p=Lf_p/Lp_p
    Lf_p = L_mtx(ny * (horizon + 1) + 1:2 * ny * horizon, :);
    Lp_p = L_mtx(1:ny * (horizon + 1), :);
    Ob_p = Lf_p * Lp_p' * pinv(Lp_p * Lp_p') * Lp_p;

    % **************************************
    %               Algorithm 1 - STEP 4 - Yang et al 2020
    % **************************************
    % Compute the SVD of Ob matrix
    [U, S, V] = svd(Ob);
    sv = diag(S);
    clear V S WOW;
    U1 = U(:, 1:nx);
    % finding the closest U1 at time step t, to the U1 at time step t-1
    % (Appendix C, Yang et al 2020)
    for j = 1:nx

        if dot(U1(:, j), U1_prev(:, j)) < 0
            U1(:, j) = -U1(:, j);

        end

    end

    U1_current = U1;
    % **************************************
    %                Algorithm 1 - STEP 5 - Yang et al 2020
    % **************************************

    % Determine the observability matrix H
    H = U1 * diag(sqrt(sv(1:nx)));
    H_inv = pinv(H);
    % Determine the observability matrix without the last ny rows Hp: 
    Hp = U1(1:ny * (horizon - 1), :) * diag(sqrt(sv(1:nx)));
    Hp_inv = pinv(Hp);

    % **************************************
    %                Algorithm 1 - STEP 6 - Yang et al 2020
    % **************************************

    % Estimate A from the observability matrix
    A = pinv(Hp) * H(ny + 1:end, :);

    a_eig_abs = abs(eig(A));

    % To ensure stability of eigenvalues of A (Appendix D, Yang et al 2020)
    if sum(a_eig_abs >= 1) > 0
        A = pinv(H) * [H(ny + 1:end, :); zeros(ny, nx)];
    end

    % **************************************
    %                Algorithm 1 - STEP 7 - Yang et al 2020
    % **************************************
    % Estimate C from the observability matrix
    C = H(1:ny, :);

    % **************************************
    %                Algorithm 1 - STEP 8 - Yang et al 2020
    % **************************************

    % Determine the states Xi and Xip
    Xi = H_inv * Ob;
    Xip = Hp_inv * Ob_p;
    clear Hp_inv

    % **************************************
    %           Algorithm 1 - STEP 9 - Yang et al 2020
    % **************************************
    % Determine residue vectors
    Rhs = [Xi];
    Lhs = [Xip; L_mtx(ny * horizon + 1:ny * (horizon + 1), :)];
    res = Lhs - [A; squeeze(C)] * Rhs;

    % **************************************
    %           Algorithm 1 - STEP 10 - Yang et al 2020
    % **************************************

    % Determine noise covariances from the residue vectors
    if beta == 1
        Cov_RQS = (res * res') / t;
    else
        Cov_RQS = (res * res') * (1 - beta) / (1 - beta^t);
    end

    %***************************************

    Q = Cov_RQS(1:nx, 1:nx);
    R = Cov_RQS(nx + 1:nx + ny, nx + 1:nx + ny);
    S = Cov_RQS(nx + 1:nx + ny, 1:nx)';
    % Compute Kalman gain (K) in equation 16 (Yang et al 2020), by solving the algebraic Riccati equation
    K = computeKalmanParamsFromQRS(A, C, Q, R, S);
    % Store model parameters in the sys struct
    sys.A = A;
    sys.C = C;
    sys.K = K;
    sys.NoiseCovariance = [Q, S; S', R];

end
