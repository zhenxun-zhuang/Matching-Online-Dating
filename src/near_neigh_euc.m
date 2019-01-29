% Nearest neighbor method using Euclidean metric
%   X_tr is a N_tr*D matrix of all training samples
%   y_tr is a N*1 vector of labels for training samples
%   X_tr is a N_te*D matrix of all test samples
%   y_te is a N*1 vector of labels for test samples
%   k is the number of clusters
function AUC = near_neigh_euc(X_tr, X_te, y_tr, y_te, k)
    m_te = size(X_te, 1);
    pred = zeros(m_te, 1);
    for i = 1:m_te
        xte = X_te(i, :);
        diff = X_tr - xte;
        dist = sum(diff.^2, 2);
        [~,I] = sort(dist, 'descend');
        pred(i) = sign(mean(y_tr(I(1:k))));
    end
    [~,~,~,AUC] = perfcurve(y_te, pred, '1');
end
