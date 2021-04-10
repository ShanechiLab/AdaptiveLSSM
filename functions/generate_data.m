% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Maryam Shanechi
% Shanechi Lab, University of Southern California
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function generates brain network activity from a linear state space model (LSSM)
% with time-varying A according to:
%                           x(t+1)=A(t)x(t)+w;
%                           y(t)=Cx(t)+v;
%                           with E{[w,v][w' v']}
%                           =[Q,S;S',R]
%                           =NoiseCovariance
% Inputs:
%     - (1) sys_iterative: a cell array of size T containing the
%     time-varying LSSM parameters during a trial.
%     - (2) x_initial: state of the LSSM at the initial time step.
% Outputs:
%     - (1) y: generated brain network activity. A matrix with size ny by
%     T, where T is the trial duration.
%     - (2) x_last: state of the LSSM at the last time-step T.

function [y, x_last] = generate_data(sys_iterative, x_initial)
    nx = size(sys_iterative{1, 1}.A, 1);
    ny = size(sys_iterative{1, 1}.C, 1);

    if isempty(x_initial)
        x(:, 1) = zeros(nx, 1);
    else
        x(:, 1) = x_initial;
    end

    NoiseCovariance = sys_iterative{1, 1}.NoiseCovariance;
    C = sys_iterative{1, 1}.C;
    y = zeros(ny, size(sys_iterative, 1));

    for t = 1:size(sys_iterative, 1)
        noise = mvnrnd(zeros(nx + ny, 1), NoiseCovariance)';
        w = noise(1:nx);
        v = noise(nx + 1:end);
        x(:, t + 1) = sys_iterative{t, 1}.A * x(:, t) + w;
        y(:, t) = C * x(:, t) + v;
    end

    x_last = x(:, end);
end
