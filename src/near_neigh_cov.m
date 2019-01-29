% Nearest neighbor method using convariance metric
%   X_tr is a N_tr*D matrix of all training samples
%   y_tr is a N*1 vector of labels for training samples
%   X_tr is a N_te*D matrix of all test samples
%   y_te is a N*1 vector of labels for test samples
%   k is the number of clusters
function AUC = near_neigh_cov(X_tr, X_te, y_tr, y_te, k)
    X_tr = X_tr - mean(X_tr, 2);
    X_te = X_te - mean(X_te, 2);

    m_te = size(X_te, 1);
    pred = zeros(m_te, 1);
    
    batch_size = 500;
    num_batch = ceil(m_te/batch_size);
    tic;
    for i = 1:num_batch
        if(i ~= num_batch)
            ind_cur = (i-1)*batch_size+1 : i*batch_size;
        else
            ind_cur = (i-1)*batch_size+1 : m_te;
        end   
        xte = X_te(ind_cur,:)';
        cov = X_tr*xte;
        [~,I] = sort(cov, 'descend');
        kmax = I(1:k, :);
        pred(ind_cur) = sign(mean(y_tr(kmax), 1));
        fprintf('Train: Batch: %d, Elapsed: %.2f\n', i, toc);
    end
    pred(pred == 0) = 1;
    [~,~,~,AUC] = perfcurve(y_te, pred, '1');
end