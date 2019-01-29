% Calculate Area Under Curve score for a linear model
% but with features expanded to the second order like x.^2
function test_auc_sq(X_te, W)
    dim = size(X_te, 2); %dimension
    
    dim_total = 1 + dim + dim*(dim+1)/2;
    n_per_batch = 20000;
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
    
    fid = fopen(strcat('../output/lin_reg_sq_lasso.csv'),'w'); 
    fprintf(fid,'%s\n', 'Id,Prediction');
    fprintf(fid,'%d,%f\n', y_te');
    fclose(fid);    
end