% Perceptron algorithm
function perceptron(X_tr, y_tr, X_te)
    m_tr = size(X_tr, 1); % Number of training samples
    d = size(X_tr, 2) + 1; % Dimension considering one additional for bias
    wi = zeros(d, 1); % Initialization
    X_tr = [X_tr, ones(m_tr,1)];
    ite_num = 2;
    for ite = 1:ite_num
        mistakes = 0;
        for i = 1:m_tr
            yi = y_tr(i);
            xi = X_tr(i, :);
            if(yi*(xi*wi) <= 0)% When a mistake happens
                wi = wi + yi*xi';
                mistakes = mistakes + 1;
            end
        end
        fprintf('Round %d, Accuracy: %f\n', ite, 1-mistakes/m_tr);
    end
    
    m_te = size(X_te, 1);
    X_te = [X_te, ones(m_te, 1)]; % Homogenize
    pred(wi, X_te, 'perceptron');     
end