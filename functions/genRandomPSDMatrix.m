% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Omid Sani
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%genRandomPSDMatrix Generates a random positive semi-definite square
%matrix
%   Inputs:
%     - (1) n: matrix dimension. Will be nxn
%     - (2) eigVals (optional): eigenvalues of the matrix. Will be abs
%               value of these. If not provided, eig vals will be drawn
%               from a random unit variance, zero mean gaussian and abs.
%               If scalar, they will be drawn from gaussian with that
%               variance.
%   Outputs:
%     - (1) M: The matrix
%   Usage example:
%       M = genRandomPosSemiDefMatrix(10);
%       M2 = genRandomPosSemiDefMatrix(10, [1:10]);

function M = genRandomPSDMatrix(n, eigVals)

    if nargin < 2
        eigVals = randn(n, 1);
    else

        if length(eigVals) == 1
            eigVals = eigVals * randn(n, 1);
        else

            if (length(eigVals) ~= n)
                error('eigVals must have length of n or be scaler!\n');
            end

        end

    end

    A = randn(n);
    [U, ~] = eig((A + A') / 2);
    M = U * diag(abs(eigVals)) * U';
end
