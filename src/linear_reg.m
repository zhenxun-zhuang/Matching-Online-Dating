% The linear regression method y = <w, x> + bias
% Input:
%   X_tr is a N_tr*D matrix of all training samples
%   y_tr is a N*1 vector of labels for training samples
%   bias denotes whether we use the bias in the model
%   X_te is a N_te*D matrix of all test samples
function [W, AUC] = linear_reg(X_tr, y_tr, bias, X_te)
    nSample = size(X_tr, 1); % Number of training samples
    dim = size(X_tr, 2); % Dimension

    if(bias)
        X_tr = [X_tr, ones(nSample, 1)]; % Homogenize
        dim = dim + 1;
    end

    % Solve Aw=B
    A = X_tr'*X_tr;
    B = X_tr'*y_tr;
    rA = rank(A);
    if(rA == dim) % Inverible
        W = A\B;
    else
        [V, D] = eig(A);
        for i = 1:dim % Since eigenvalues in D are increasingly ordered
            if(D(i, i) > 1e-4) % Eliminate those tiny eigenvalues that are close to 0
                D(i, i) = 1 / D(i, i);
            else
                D(i, i) = 0;
            end
        end
        W = V*D*V'*B;
    end
    
    [~,~,~,AUC] = perfcurve(y_tr, X_tr*W, '1');
    
    m_te = size(X_te, 1);
    if(bias)
        X_te = [X_te, ones(m_te, 1)]; % Homogenize
    end
    test_pred(W, X_te, 'linear_reg');
end