% Primal form Support Vector Machine optimized by Gradient Descent
function wbar = svm_primal_gd(X_tr, y_tr, C, X_te)
    m_tr = size(X_tr, 1);%number of samples
    d = size(X_tr, 2); %dimension    
    w = linear_reg(X_tr, y_tr, 1, X_te);
    X_tr = [X_tr, ones(m_tr, 1)];
    wsum = zeros(d+1, 1);
    max_ite = 100;
    a = 0.01;
    for ite = 1:max_ite
        pred = (X_tr*w).*y_tr;
        grad = y_tr.*X_tr;
        grad = mean(grad(pred<1,:));
        w = w - (a/ite)*(w - C*grad');
        wsum = wsum + w;
        wbar = wsum / ite;
        fprintf('Iteration: %d, AUC: %.4f\n', ite, train_auc(X_tr, y_tr, wbar));
    end
    
    m_te = size(X_te, 1);
    X_te = [X_te, ones(m_te, 1)]; % Homogenize
    test_pred(wbar, X_te, 'svm_primal');
end