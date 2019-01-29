% Calculate Area Under Curve score for a linear model
function AUC = train_auc(X_tr, y_tr, w)
    scores = X_tr*w;
    [~,~,~,AUC] = perfcurve(y_tr,scores,'1');
end