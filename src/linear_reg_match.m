% Similar to linear regression but we generate more features by multiplying
% the first half of each sample by its second half. The reason is that a 
% sample contains two profiles, so the correlation of same features between
% two people should be useful.
function [W, y_te, AUC] = linear_reg_match(X_tr, y_tr, X_va, y_va, X_te, test)
    AUC = zeros(1,2);
    
    poly_each = 4; % Used to denote the maximum power of each dimension
    poly_pair = 2; % Used to denote the maximum power of each paired dimension
    
    dim = size(X_tr, 2); % Dimension
    dim_half = dim/2; % Half of dimension used to form pairs
    
    n_per_batch = 20000; % Number of samples used in each batch
    
    for d2 = 1
        fprintf('D2: %d\n', d2);
        
        % Total number of dimensions in the extended version
        dim_total = 1 + poly_each*dim + poly_pair*(dim_half-1) + dim;

        %--------------------------------train--------------------------------
        m_tr = size(X_tr, 1); % Number of training samples
        n_batch = ceil(m_tr/n_per_batch); % Number of training batches
        
        A = zeros(dim_total, dim_total);
        B = zeros(dim_total, 1);
        
        tic;
        for k = 1:n_batch
            if(k ~= n_batch)
                xk = X_tr((k-1)*n_per_batch+1 : k*n_per_batch , :);
                yk = y_tr((k-1)*n_per_batch+1 : k*n_per_batch);
            else
                xk = X_tr((k-1)*n_per_batch+1 : end , :);
                yk = y_tr((k-1)*n_per_batch+1 : end);
            end

            xk_ex = xk_calc(xk, d2, dim, dim_half, dim_total, poly_each, poly_pair);

            A = A + xk_ex'*xk_ex;
            B = B + xk_ex'*yk;

            fprintf('Train: Batch: %d/%d, Elapsed: %.2f\n', k, n_batch, toc);        
        end

        rA = rank(A);
        if(rA == dim_total) % Inverible
            W = A\B;
        else % Non-invertible
            [V, D] = eig(A);
            for i = 1:dim_total % Since eigenvalues in D are ascending
                if(D(i, i)>1e-4) % Consider those tiny eigenvalues as 0
                    D(i, i) = 1/D(i, i);
                else
                    D(i, i) = 0;
                end
            end
            W = V*D*V'*B;
        end

        %-----------------------------AUC of train----------------------------
        pred = zeros(m_tr, 1); % Prediction values of the training set
        n_batch = ceil(m_tr/n_per_batch);
        tic;
        for batch = 1:n_batch
            if(batch ~= n_batch)
                xk = X_tr((batch-1)*n_per_batch+1 : batch*n_per_batch , :);
            else
                xk = X_tr((batch-1)*n_per_batch+1 : end , :);
            end

            xk_ex = xk_calc(xk, d2, dim, dim_half, dim_total, poly_each, poly_pair);
            
            if(batch ~= n_batch)
                pred((batch-1)*n_per_batch+1 : batch*n_per_batch) = xk_ex*W;
            else
                pred((batch-1)*n_per_batch+1 : end) = xk_ex*W;
            end

            fprintf('Train AUC: Batch: %d/%d, Elapsed: %.2f\n', batch, n_batch, toc);
        end
        [~,~,~,AUC_tr] = perfcurve(y_tr,pred,'1');
        fprintf('AUC Train: %.4f\n', AUC_tr);    

        %------------------------------validation-----------------------------
        m_va = size(X_va, 1);
        prva = zeros(m_va, 1); % Prediction values of the validation set
        n_batch = ceil(m_va/n_per_batch);
        tic;
        for batch = 1:n_batch
            if(batch ~= n_batch)
                xk = X_va((batch-1)*n_per_batch+1 : batch*n_per_batch , :);
            else
                xk = X_va((batch-1)*n_per_batch+1 : end , :);
            end

            xk_ex = xk_calc(xk, d2, dim, dim_half, dim_total, poly_each, poly_pair);
            
            if(batch ~= n_batch)
                prva((batch-1)*n_per_batch+1 : batch*n_per_batch) = xk_ex*W;
            else
                prva((batch-1)*n_per_batch+1 : end) = xk_ex*W;
            end

            fprintf('Val AUC: Batch: %d/%d, Elapsed: %.2f\n', batch, n_batch, toc);
        end
        [~,~,~,AUC_va] = perfcurve(y_va,prva,'1');
        fprintf('AUC Validation: %.4f\n', AUC_va);
        
        AUC(d2, 1) = AUC_tr;
        AUC(d2, 2) = AUC_va;
    end
    
    m_te = size(X_te, 1);
    y_te = zeros(m_te, 2); % Prediction values of the test set
    %------------------------------test-----------------------------
    if(test)
        n_batch = ceil(m_te/n_per_batch);
        for k = 1:n_batch
            if(k ~= n_batch)
                xk = X_te((k-1)*n_per_batch+1 : k*n_per_batch , :);
            else
                xk = X_te((k-1)*n_per_batch+1 : end , :);
            end        

            xk_ex = xk_calc(xk, d2, dim, dim_half, dim_total, poly_each, poly_pair);     

            if(k ~= n_batch)
                y_te((k-1)*n_per_batch+1 : k*n_per_batch, :) = [((k-1)*n_per_batch+1 : k*n_per_batch)',xk_ex*W];
            else
                y_te((k-1)*n_per_batch+1 : end , :) = [((k-1)*n_per_batch+1 : m_te)', xk_ex*W];
            end
        end

        fid = fopen(strcat('../output/lin_reg_match.csv'),'w'); 
        fprintf(fid,'%s\n', 'Id,Prediction');
        fprintf(fid,'%d,%f\n', y_te');
        fclose(fid);
    end
end

% Calculate the extended version of xk
function xk_ex = xk_calc(xk, dim, dim_half, dim_total, poly_each, poly_pair)
    m_cur = size(xk, 1); % Number of samples in current batch

    xk_ex = zeros(m_cur, dim_total); % Initialization

    xk_ex(:, 1) = ones(m_cur, 1); % Constant

    for i = 1:poly_each % Polynomial of each dimension
        xk_ex(:, ((i-1)*dim+2):(i*dim+1)) = xk.^i;
    end

    ind = poly_each*dim + 2; % Correlation of each dimension
    for i = 1:dim
        xk_ex(:, ind) =  xk(:, 1).*xk(:, i);
        ind = ind + 1;
    end

    half1 = xk(:, 2:dim_half);
    half2 = xk(:, (dim_half+2):end);
    
    for po2 = 1:poly_pair % Polynomial of each paired dimension
        xk_ex(:, ind:(ind+dim_half-2)) = (half1.*half2).^(2*po2-1);
        ind = ind + dim_half - 1;
    end
end