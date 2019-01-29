% The linear regression method y = <w, x> + bias
% but with x generating third order features like x.^3
% Input:
%   X_tr is a N_tr*D matrix of all training samples
%   y_tr is a N_tr*1 vector of labels for training samples
%   X_va is a N_va*D matrix of all validation samples
%   y_va is a N_va*1 vector of labels for validation samples
%   X_te is a N_te*D matrix of all test samples
function [W, y_te, AUC_tr, AUC_va] = linear_reg_cube(X_tr, y_tr, X_va, y_va, X_te)
    m_tr = size(X_tr, 1);
    dim = size(X_tr, 2);
    d1 = dim;
    d2 = 85;
    d3 = 0; %number of components used to calc the third order
    dim_total = 1 + d1 + d2*(d2+1)/2 + d3*(d3+1)*(d3+2)/6;

    A = zeros(dim_total, dim_total);
    B = zeros(dim_total, 1);
    n_per_batch = 10000;
    n_batch = ceil(m_tr/n_per_batch);
    tic;
    for batch = 1:n_batch
        if(batch ~= n_batch)
            x_cur = X_tr((batch-1)*n_per_batch+1 : batch*n_per_batch , :);
            y_cur = y_tr((batch-1)*n_per_batch+1 : batch*n_per_batch);
        else
            x_cur = X_tr((batch-1)*n_per_batch+1 : end , :);
            y_cur = y_tr((batch-1)*n_per_batch+1 : end);
        end
        m_cur = size(x_cur, 1);
        xcur_ex = zeros(m_cur, dim_total);
        
        xcur_ex(:, 1) = ones(m_cur, 1);
        
        xcur_ex(:, 2:(1+d1)) = x_cur(:,1:d1);
        
        ind = 1 + d1 + 1;
        for i = 1:d2
            for j = i:d2
                xcur_ex(:, ind) =  xcur_ex(:, i).*xcur_ex(:, j);
                ind = ind + 1;
            end
        end
        
        ind = 1 + d1 + d2*(d2+1)/2 + 1;
        for i = 1:d3
            for j = i:d3
                for k = j:d3
                    xcur_ex(:, ind) =  xcur_ex(:, i).*xcur_ex(:, j).*xcur_ex(:, k);
                    ind = ind + 1;                    
                end
            end
        end
        
        A = A + xcur_ex'*xcur_ex;
        B = B + xcur_ex'*y_cur;
        
        fprintf('Train: Batch: %d/%d, Elapsed: %.2f\n', batch, n_batch, toc);
    end

    rA = rank(A);
    if(rA == dim_total) % Inverible
        W = A\B;
    else
        [V, D] = eig(A);
        for i = 1:dim_total % Since eigenvalues in D are increasingly ordered
            if(D(i, i)>1e-4) % Eliminate those tiny eigenvalues that are actually 0
                D(i, i) = 1/D(i, i);
            else
                D(i, i) = 0;
            end
        end
        W = V*D*V'*B;
    end

    
    %train_auc
    pred = zeros(m_tr, 1);
    n_batch = ceil(m_tr/n_per_batch);
    tic;
    for batch = 1:n_batch
        if(batch ~= n_batch)
            x_cur = X_tr((batch-1)*n_per_batch+1 : batch*n_per_batch , :);
        else
            x_cur = X_tr((batch-1)*n_per_batch+1 : end , :);
        end
        m_cur = size(x_cur, 1);
        xcur_ex = zeros(m_cur, dim_total);        
        
        xcur_ex(:, 1) = ones(m_cur, 1);
        
        xcur_ex(:, 2:(1+d1)) = x_cur(:,1:d1);
        
        ind = 1 + d1 + 1;
        for i = 1:d2
            for j = i:d2
                xcur_ex(:, ind) =  xcur_ex(:, i).*xcur_ex(:, j);
                ind = ind + 1;
            end
        end
        
        ind = 1 + d1 + d2*(d2+1)/2 + 1;
        for i = 1:d3
            for j = i:d3
                for k = j:d3
                    xcur_ex(:, ind) =  xcur_ex(:, i).*xcur_ex(:, j).*xcur_ex(:, k);
                    ind = ind + 1;                    
                end
            end
        end        
        
        if(batch ~= n_batch)
            pred((batch-1)*n_per_batch+1 : batch*n_per_batch) = xcur_ex*W;
        else
            pred((batch-1)*n_per_batch+1 : end) = xcur_ex*W;
        end
        
        fprintf('Train AUC: Batch: %d/%d, Elapsed: %.2f\n', batch, n_batch, toc);
    end
    [~,~,~,AUC_tr] = perfcurve(y_tr,pred,'1');
    fprintf('AUC Train: %.4f\n', AUC_tr);
    
    %val_auc
    m_va = size(X_va, 1);
    prva = zeros(m_va, 1);
    n_batch = ceil(m_va/n_per_batch);
    tic;
    for batch = 1:0
        if(batch ~= n_batch)
            x_cur = X_va((batch-1)*n_per_batch+1 : batch*n_per_batch , :);
        else
            x_cur = X_va((batch-1)*n_per_batch+1 : end , :);
        end
        m_cur = size(x_cur, 1);
        xcur_ex = zeros(m_cur, dim_total);        
        
        xcur_ex(:, 1) = ones(m_cur, 1);
        
        xcur_ex(:, 2:(1+d1)) = x_cur(:,1:d1);
        
        ind = 1 + d1 + 1;
        for i = 1:d2
            for j = i:d2
                xcur_ex(:, ind) =  xcur_ex(:, i).*xcur_ex(:, j);
                ind = ind + 1;
            end
        end
        
        ind = 1 + d1 + d2*(d2+1)/2 + 1;
        for i = 1:d3
            for j = i:d3
                for k = j:d3
                    xcur_ex(:, ind) =  xcur_ex(:, i).*xcur_ex(:, j).*xcur_ex(:, k);
                    ind = ind + 1;                    
                end
            end
        end        
        
        if(batch ~= n_batch)
            prva((batch-1)*n_per_batch+1 : batch*n_per_batch) = xcur_ex*W;
        else
            prva((batch-1)*n_per_batch+1 : end) = xcur_ex*W;
        end
        
        fprintf('Val AUC: Batch: %d/%d, Elapsed: %.2f\n', batch, n_batch, toc);
    end
    [~,~,~,AUC_va] = perfcurve(y_va,prva,'1');
    fprintf('AUC Validation: %.4f\n', AUC_va);
    
    
    %test    
    m_te = size(X_te, 1);
    y_te = zeros(m_te, 2);

    n_batch = ceil(m_te/n_per_batch);
    tic;
    for batch = 1:n_batch
        if(batch ~= n_batch)
            x_cur = X_te((batch-1)*n_per_batch+1 : batch*n_per_batch , :);
        else
            x_cur = X_te((batch-1)*n_per_batch+1 : end , :);
        end
        m_cur = size(x_cur, 1);
        xcur_ex = zeros(m_cur, dim_total);        
        
        xcur_ex(:, 1) = ones(m_cur, 1);
        
        xcur_ex(:, 2:(1+d1)) = x_cur(:,1:d1);
        
        ind = 1 + d1 + 1;
        for i = 1:d2
            for j = i:d2
                xcur_ex(:, ind) =  xcur_ex(:, i).*xcur_ex(:, j);
                ind = ind + 1;
            end
        end
        
        ind = 1 + d1 + d2*(d2+1)/2 + 1;
        for i = 1:d3
            for j = i:d3
                for k = j:d3
                    xcur_ex(:, ind) =  xcur_ex(:, i).*xcur_ex(:, j).*xcur_ex(:, k);
                    ind = ind + 1;                    
                end
            end
        end        
        
        if(batch ~= n_batch)
            y_te((batch-1)*n_per_batch+1 : batch*n_per_batch, :) = [((batch-1)*n_per_batch+1 : batch*n_per_batch)',xcur_ex*W];
        else
            y_te((batch-1)*n_per_batch+1 : end , :) = [((batch-1)*n_per_batch+1 : m_te)', xcur_ex*W];
        end
        
        fprintf('Test: Batch: %d/%d, Elapsed: %.2f\n', batch, n_batch, toc);
    end

    fid = fopen(strcat('../output/lin_reg_sq85.csv'),'w'); 
    fprintf(fid,'%s\n', 'Id,Prediction');
    fprintf(fid,'%d,%f\n', y_te');
    fclose(fid);
end