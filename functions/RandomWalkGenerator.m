% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Maryam Shanechi
% Shanechi Lab, University of Southern California
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function generates a random walk (rw) by filtering a white Gaussian noise
% using a bandpass filter with cutoff frequency (FC1,FC1+eps). FC1 determines
% how fast eigenvalue time-series change over time.
% Inputs:
%     - (1) T: length of the random walk
%     - (2) speed: speed of change of the random walk
%     - (3) n_rw: number of random walk time-series
% Outputs:
%     - (1) rw: generated random walk
% Refer to Yang et al 2020, Section 4.7 for details.

function [rw] = RandomWalkGenerator(T, speed, n_rw)
    delete_t = 5 * fix(1 / speed); % segment length to delete after filtering
    raw_random_signal = randn(n_rw, T + delete_t);
    %% Design a filter with desired cutoff frequencies to filter
    %% raw_random_signal with it.
    Fs = 1; % Sampling Frequency
    N = 8; % Order of the filter
    Fc1 = speed; % First Cutoff Frequency
    Fc2 = Fc1 + 0.0002; % Second Cutoff Frequency
    % Construct an FDESIGN object and call its BUTTER method.
    h = fdesign.bandpass('N,F3dB1,F3dB2', 8, Fc1, Fc2, Fs);
    Hd = design(h, 'butter');
    %% Filtering A_fluc_temp
    [rw_t] = filter(Hd, raw_random_signal');
    rw = rw_t';
    clear raw_random_signal;
    %% Scale rw to be between -1 and 1 at minimum and maximum.

    rw = rw(:, delete_t + 1:end);
    max_rw = max(rw, [], 2);
    min_rw = min(rw, [], 2);
    range_rw = max_rw - min_rw;
    mid_rw = (max_rw + min_rw) / 2;
    rw = 2 * (rw - mid_rw) ./ range_rw;
end
