% The ridge regression method y = <w, x> + bias
% Input:
%   X_tr is a N_tr*D matrix of all training samples
%   y_tr is a N*1 vector of labels for training samples
%   lambda controls the effects of regularization
%   X_te is a N_te*D matrix of all test samples
function ridge_reg(X_tr, y_tr, lambda, X_te)
    m_tr = size(X_tr,1); % Number of samples
    d = size(X_tr,2); % Dimension
    Xbar = [X_tr, ones(m_tr, 1)]';
    Ibar = [eye(d), zeros(d,1); zeros(1,d), 0];
    C = Xbar*Xbar' + lambda*Ibar;
    W = C\(Xbar*y_tr);
    
    m_te = size(X_te, 1);
    X_te = [X_te, ones(m_te, 1)]; % Homogenize
    pred(W, X_te, 'ridge_reg');    
end