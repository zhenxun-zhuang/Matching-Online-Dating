% Used to compute the predicted labels for the Linear Regression Method
function test_pred(w, X_te, method)
    y_te = X_te*w;
    m = size(y_te, 1);
    y_te = [(1:m)', y_te];
    
    fid = fopen(strcat('../output/', method, '.csv'),'w'); 
    fprintf(fid,'%s\n', 'Id,Prediction');
    fprintf(fid,'%d,%f\n', y_te');
    fclose(fid);
end