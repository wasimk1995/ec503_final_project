function Y_pred = moe_test(X_test,MOE_model,numofClass,k)
    %p(y|x,theda)
    W = MOE_model.W;
    V = MOE_model.V;
    
    n = size(X_test,1);
    d = size(X_test,2)+1;
    m = numofClass;
    
    X_test_ext = [X_test,ones(n,1)];
    pi_i_k = reshape(repmat(exp(X_test_ext*V)./(sum(exp(X_test_ext*V),2)*ones(1,k)),m,1),n,[]);
    p_n_k = exp(X_test_ext*W)./reshape(repmat(sum(reshape(exp(X_test_ext*W),n,m,k),3),1,k),n,[]);
    pred = sum(reshape(pi_i_k.*p_n_k,n,m,k),3);
    [~,Y_pred] = max(pred,[],2);
    Y_pred = Y_pred - 1;
end