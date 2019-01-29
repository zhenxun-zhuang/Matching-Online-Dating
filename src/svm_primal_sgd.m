% Primal form Support Vector Machine optimized by SGD
function wbar = svm_primal_sgd(X_tr, y_tr, C, X_te)
    m_tr = size(X_tr, 1);% Number of samples
    d = size(X_tr, 2); % Dimension    
    v = linear_reg(X_tr, y_tr, 1, X_te);
    X_tr = [X_tr, ones(m_tr, 1)];
    wsum = zeros(d+1, 1);
    max_ite = 100;
    for ite = 1:max_ite
        ind = randperm(m_tr);
        for i = 1:m_tr
            x = X_tr(ind(i),:);
            y = y_tr(ind(i));
            w = v./(C*i);
            wsum = wsum + w;
            if y*(x*w) < 1
                v = v + y*x';
            end
        end
        wbar = wsum./(m_tr*ite);
        fprintf('Iteration: %d, AUC: %.4f\n', ite, train_auc(X_tr, y_tr, wbar));
    end
    wbar = wsum./(m_tr*max_ite);
    
    m_te = size(X_te, 1);
    X_te = [X_te, ones(m_te, 1)]; % Homogenize
    test_pred(wbar, X_te, 'svm_primal');
end
