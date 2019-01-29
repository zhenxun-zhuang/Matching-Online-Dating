% Kernal SVM Method in Dual Form
% Input:
%   X_tr is a N_tr*D matrix of all training samples
%   y_tr is a N*1 vector of labels for training samples
%   C is the inverse of the Lagrangian multiplier
%   kernel is the kind of kernel used
%   gamma is a parameter of the kernel
%   X_tr is a N_te*D matrix of all test samples
function ksvm_dual(X_tr, y_tr, C, kernel, gamma, X_te)
    m_tr = size(X_tr, 1);
    m_te = size(X_te, 1);

    n_per_batch = 5000; % Mini-Batch size
    n_batch_tr = ceil(m_tr/n_per_batch); % Number of batches in the training set
    n_batch_te = ceil(m_te/n_per_batch); % Number of batches in the test set
    
    alpha = zeros(m_tr, 1); % Dual variables
    ypred = zeros(m_te, 1); % Predicted labels
    tic;
    for k_tr = 1:n_batch_tr
        if(k_tr ~= n_batch_tr)
            xk_tr = X_tr((k_tr-1)*n_per_batch+1 : k_tr*n_per_batch , :);
            yk_tr = y_tr((k_tr-1)*n_per_batch+1 : k_tr*n_per_batch);
        else
            xk_tr = X_tr((k_tr-1)*n_per_batch+1 : end , :);
            yk_tr = y_tr((k_tr-1)*n_per_batch+1 : end);
        end
        m_cur_tr = size(xk_tr, 1);
        
        G = zeros(m_cur_tr,m_cur_tr); % Gram matrix
        if strcmp(kernel,'gaussian')
            for i = 1:m_cur_tr
                x_i = xk_tr(i, :);
                for j = 1:m_cur_tr
                    xd = x_i - xk_tr(j, :);
                    G(i,j) = exp(-gamma*(xd'*xd));
                end
            end
        elseif strcmp(kernel,'linear')
            G = (1 + xk_tr*xk_tr').^gamma;
        end

        % Solving the quadratic programming problem
        H = yk_tr*yk_tr'.*G;
        f = -ones(m_cur_tr,1);
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = zeros(m_cur_tr,1);
        ub = C*ones(m_cur_tr,1);

        alpha_cur = quadprog(H, f, A, b, Aeq, beq, lb, ub);
        alpha((k_tr-1)*n_per_batch+1 : (k_tr-1)*n_per_batch + m_cur_tr) = alpha_cur;
 
        % Compute the predicted label
        ypred_cur = zeros(m_te,1);
        for k_te = 1:n_batch_te
            if(k_te ~= n_batch_te)
                xk_te = X_te((k_te-1)*n_per_batch+1 : k_te*n_per_batch , :);
            else
                xk_te = X_te((k_te-1)*n_per_batch+1 : end , :);
            end
            m_cur_te = size(xk_te, 1);

            if strcmp(kernel,'linear')
                ypred_cur_k = (1 + xk_te*xk_tr').^gamma*(alpha_cur.*yk_tr);
            elseif strcmp(kernel,'gaussian')
                G = zeros(m_cur_te,m_cur_tr); %Gram matrix
                for i = 1:m_te
                    x_te = xk_te(i,:);
                    for j = 1:m_tr
                        xd = x_te - xk_tr(j,:);
                        G(i,j) = exp(-gamma*(xd'*xd));
                    end
                end
                ypred_cur_k = (G*(alpha_cur.*yk_tr));
            end
            
            if(k_te ~= n_batch_te)
                ypred_cur((k_te-1)*n_per_batch+1 : k_te*n_per_batch) = ypred_cur_k;
            else
                ypred_cur((k_te-1)*n_per_batch+1 : end) = ypred_cur_k;
            end           
        end
        
        ypred = ypred + ypred_cur*(m_cur_tr/n_per_batch);
        
        time_elapsed = toc;
        fprintf('Round %d, Total %f, Left %f\n', k_tr, time_elapsed/k_tr*n_batch_tr, time_elapsed/k_tr*n_batch_tr - time_elapsed);
    end
    
    ypred = ypred/n_batch_tr;
    ypred = [(1:m_te)', ypred];
    fid = fopen(strcat('../output/',kernel,'_kernel_svm.csv'),'w'); 
    fprintf(fid,'%s\n', 'Id,Prediction');
    fprintf(fid,'%d,%f\n', ypred');
    fclose(fid);    
end
