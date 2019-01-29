% K-Nearest-Neighbor Method
m_tr = size(X_trn, 1);
ind = randperm(m_tr);
Xtr = X_trn(ind(1:300000),:);
ytr = y_tr(ind(1:300000),:);
Xte = X_trn(ind(300001:end),:);
yte = y_tr(ind(300001:end),:);
AUC = near_neigh_cov(Xtr, Xte, ytr, yte, 10);