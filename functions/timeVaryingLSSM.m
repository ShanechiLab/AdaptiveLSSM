% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Yuxiao Yang, Maryam Shanechi
% Shanechi Lab, University of Southern California
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function generates parameters of the following linear state space models (LSSMs)
% with time-varying A:
%                           x(t+1)=A(t)x(t)+w;
%                           y(t)=Cx(t)+v;
%                           with E{[w,v][w' v']}
%                           =[Q,S;S',R]
%                           =NoiseCovariance
% The magnitude and angle of the eigenvalues of A changes over time
% according to a random walk. Refer to Yang et al 2020, section 4.7 for more details.
%   Inputs:
%     - (1) nx: dimension of latent state
%     - (2) T: number of time-steps of the LSSM
%     - (3) speed: speed of change for eigenvalues of A which change over time according to a random walk.
%     - (4) amp_range: the range of change for magnitude of eigenvalues of A which
%     change over time according to a random walk.
%     - (5) angle_range: the range of change for angles of eigenvalues of A which
%     change over time according to a random walk.
%   Outputs:
%     - (1) sys_iterative: a cell array of length T containing the time-varying LSSM parameters

function [sys_iterative] = timeVaryingLSSM(nx, T, speed, amp_range, angle_range)

    n_conj_pairs = floor(nx / 2); % number of complex modes of A matrix (could be any other integer smaller than floor(nx/2) as well).
    n_modes = n_conj_pairs + nx - n_conj_pairs * 2; % number of all modes of A matrix
    ny = 2 * nx; % number of neural observations (could be anything else)
    % create random walks between -1 and 1 for duration of T, with the given speed of change.
    abs_rw_norm = RandomWalkGenerator(T, speed, n_modes); % normalized random walks for the magnitude of the eigenvalues of A
    angle_rw_norm = RandomWalkGenerator(T, speed, n_modes); % normalized random walks for the angle of the eigenvalues of A
    %%
    t = 1;

    while t ~= T
        min_offset_value = amp_range / 2 + 0.05; % We want minimum value of magnitude of eigenvalues of A to be 0.05
        max_offset_value = -amp_range / 2 + 0.95; % We want maximum value of magnitude of eigenvalues of A to be 0.95 (for stability of the system it should be between 0 and 1)
        offset_values = min_offset_value:0.05:max_offset_value; % magnitude of eigenvalues of A are  randoms walk around A_offset.

        if n_modes > length(offset_values)
            sys.A_offset = offset_values(randi(length(offset_values), 1, n_modes));
        else
            sys.A_offset = offset_values(randperm(length(offset_values), n_modes));
        end

        sys.C = randn(ny, nx);
        sys.NoiseCovariance = genRandomPSDMatrix(nx + ny);
        sys.Q = sys.NoiseCovariance(1:nx, 1:nx);
        sys.R = sys.NoiseCovariance(nx + 1:end, nx + 1:end);
        sys.S = sys.NoiseCovariance(nx + 1:end, 1:nx);

        sys_iterative = cell(T, 1); % Cell for storing system parameters at all time steps

        for t = 1:T
            sys_iterative{t, 1}.A = zeros(nx, nx);
            k = 1;

            for j = 1:n_conj_pairs
                A_abs = sys.A_offset(j) + abs_rw_norm(j, t) * amp_range / 2;
                A_angle = angle_rw_norm(j, t) * angle_range / 2 + angle_range / 2;
                temp_block = min(A_abs * [cos(A_angle), -sin(A_angle); sin(A_angle), cos(A_angle)], 0.99 * ones(2, 2));
                sys_iterative{t, 1}.A(k:k + 1, k:k + 1) = max(temp_block, -0.99 * ones(2, 2));
                k = k + 2;
            end

            for j = n_conj_pairs + 1:n_modes
                sys_iterative{t, 1}.A(k, k) = sys.A_offset(j) + abs_rw_norm(j, t) * amp_range / 2;
            end

            % checking observability and controllability criteria, if not
            % met, model parameters are regenerated!
            if (rank(obsv(sys_iterative{t, 1}.A, sys.C)) ~= nx || rank(ctrb(sys_iterative{t, 1}.A, sqrtm(sys.Q))) ~= nx)
                break
            end

        end

    end

    % Computing the Kalman gain for all time steps t
    for t = 1:T
        sys_iterative{t, 1}.C = sys.C;

    end

    sys_iterative{1, 1}.NoiseCovariance = sys.NoiseCovariance;
    sys_iterative{1, 1}.A_offsets = sys.A_offset;
end
