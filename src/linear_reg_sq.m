% The linear regression method y = <w, x> + bias
% but with x generating second order features like x.^2
% Input:
%   X_tr is a N_tr*D matrix of all training samples
%   y_tr is a N_tr*1 vector of labels for training samples
%   X_te is a N_te*D matrix of all test samples
function [W, y_te] = linear_reg_sq(X_tr, y_tr, X_te)
    m_tr = size(X_tr, 1);
    dim = size(X_tr, 2); % Dimension
    
    dim_total = 1 + dim + dim*(dim+1)/2;

    A = zeros(dim_total, dim_total);
    B = zeros(dim_total, 1);
    n_per_batch = 20000;
    n_batch = ceil(m_tr/n_per_batch);
    for k = 1:n_batch
        if(k ~= n_batch)
            xk = X_tr((k-1)*n_per_batch+1 : k*n_per_batch , :);
            yk = y_tr((k-1)*n_per_batch+1 : k*n_per_batch);
        else
            xk = X_tr((k-1)*n_per_batch+1 : end , :);
            yk = y_tr((k-1)*n_per_batch+1 : end);
        end
        m_cur = size(xk, 1);
        xk_ex = zeros(m_cur, dim_total);
        xk_ex(:, 1) = ones(m_cur, 1);
        xk_ex(:, 2:(1+dim)) = xk;
        ind = dim + 2;
        for i = 1:dim
            for j = i:dim
                xk_ex(:, ind) =  xk_ex(:, i).*xk_ex(:, j);
                ind = ind + 1;
            end
        end
        A = A + xk_ex'*xk_ex;
        B = B + xk_ex'*yk;       
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
    
    
    m_te = size(X_te, 1);
    y_te = zeros(m_te, 2);

    n_batch = ceil(m_te/n_per_batch);
    for k = 1:n_batch
        if(k ~= n_batch)
            xk = X_te((k-1)*n_per_batch+1 : k*n_per_batch , :);
        else
            xk = X_te((k-1)*n_per_batch+1 : end , :);
        end
        m_cur = size(xk, 1);
        xk_ex = zeros(m_cur, dim_total);
        xk_ex(:, 1) = ones(m_cur, 1);
        xk_ex(:, 2:(1+dim)) = xk;
        ind = dim + 2;
        for i = 1:dim
            for j = i:dim
                xk_ex(:, ind) =  xk_ex(:, i).*xk_ex(:, j);
                ind = ind + 1;
            end
        end
        
        if(k ~= n_batch)
            y_te((k-1)*n_per_batch+1 : k*n_per_batch, :) = [((k-1)*n_per_batch+1 : k*n_per_batch)',xk_ex*W];
        else
            y_te((k-1)*n_per_batch+1 : end , :) = [((k-1)*n_per_batch+1 : m_te)', xk_ex*W];
        end
    end

    fid = fopen(strcat('../output/lin_reg_sq2.csv'),'w'); 
    fprintf(fid,'%s\n', 'Id,Prediction');
    fprintf(fid,'%d,%f\n', y_te');
    fclose(fid);
end