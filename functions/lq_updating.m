% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Parima Ahmadipour, Maryam Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function updates the L component of LQ decomposition of an arbitrary matrix M,
% when one new column is added to its right and the rest is multiplied with sqrt(beta).
% Inputs:
%     - (1) L: L component in LQ decomposition of M
%     - (2) y_new: The new column added to sqrt(beta)M
%     - (3) beta: scaling factor of M. Forgetting factor in the Adaptive
%     LSSM fitting algorithm.
%   Outputs:
%     - (1) L_new: updated L component in LQ decomposition after adding
%     y_new column to sqrt(beta)M!

function [L_new] = lq_updating(L, y_new, beta);
    % updating the L in LQ decomposition of M (arbitrary matrix) when new column is added to it.
    % M=[L,0]*Q', M'=Q*[R;0] where R=L';
    R = L';
    j = size(R, 1) +1;
    dummyQ = eye(size(R, 1));
    [~, R_new] = qrinsert(dummyQ, sqrt(beta) * R, j, y_new', 'row'); % updating R when a new row is added to sqrt(beta)M'
    % Note that if we provide the qrinsert function with the true Q component (1st input), we can get
    % updated Q as well. However, here we just provide a dummy unitary matrix (dummyQ), since we do
    % not need to keep track of the updated Q.
    L_new = R_new'; % updating L when a new column is added to data

end
