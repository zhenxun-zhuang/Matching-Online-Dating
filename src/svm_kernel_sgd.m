% Kernel SVM optimized by SGD
function svm_kernel_sgd(X_tr, y_tr, lambda, X_te)
    m_tr = size(X_tr, 1); % Number of samples

    beta = zeros(m_tr, 1);
    alpha_sum = zeros(m_tr, 1);
    
    max_ite = 100;
    batch_size = 2000;
    num_batch = ceil(m_tr/batch_size);
    tic;
    for ite = 1:max_ite
        ind = randperm(m_tr);
        for k = 1:num_batch
            if(k ~= num_batch)
                ind_cur = ind((k-1)*batch_size+1 : k*batch_size);
            else
                ind_cur = ind((k-1)*batch_size+1 : end);
            end
            
            x = X_tr(ind_cur , :)';
            y = y_tr(ind_cur);
            
            alpha = beta/(lambda*((ite-1)*num_batch+k));
            alpha_sum = alpha_sum + alpha;
            
            inne = y.*(alpha'*((X_tr*x+1).^2))';
            beta(ind_cur) = beta(ind_cur) + (y.*(inne<1));
            fprintf('Train: Iteration: %d, Batch: %d, Elapsed: %.2f\n', ite, k, toc);
        end
        alpha_bar = alpha_sum / (ite*m_tr);
        
        scores = zeros(m_tr, 1);
        for k = 1:num_batch
            if(k ~= num_batch)
                ind_cur = (k-1)*batch_size+1 : k*batch_size;
            else
                ind_cur = (k-1)*batch_size+1 : m_tr;
            end
            x = X_tr(ind_cur , :)';
            scores(ind_cur) = (alpha_bar'*((X_tr*x+1).^2))';
            fprintf('AUC: Iteration: %d, Batch: %d, Elapsed: %.2f\n', ite, k, toc);
        end
        [~,~,~,AUC] = perfcurve(y_tr,scores,'1');
        fprintf('Iteration: %d, AUC: %.4f\n', ite, AUC);
    end

    test_pred(wbar, X_te, 'svm_primal');
end
