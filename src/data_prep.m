X_tr = csvread('../data/train.csv');
y_tr = X_tr(:, end);
X_tr = X_tr(:, 1:(end-1));
X_te = csvread('../data/test.csv');
X_te = X_te(:, 1:(end-1));
sup = 1; % Desired maximum
inf = -1; % Desired minimum
[X_trn, X_ten] = normalizeAll(X_tr, X_te);
save('../data/X_trn.mat','X_trn');
save('../data/y_tr.mat' ,'y_tr' );
save('../data/X_ten.mat','X_ten');

% Use a linear model x' = ax + b to normalize all training samples
% and test sample to  the range [inf, sup]
function [X_train_norm, X_test_norm] = normalizeAll(X_train, X_test, inf, sup)
    X_train_norm = zeros(size(X_train));
    X_test_norm = zeros(size(X_test));
    for i = 1:size(X_train, 2) % Executed for each feature
        xMax = max(X_train(:, i));
        xMin = min(X_train(:, i));
        a = (sup - inf)/(xMax - xMin);
        b = inf - a.*xMin;
        X_train_norm(:, i) = a.*X_train(:, i) + b;
        X_test_norm(:, i) = a.*X_test(:, i) + b;
    end
end