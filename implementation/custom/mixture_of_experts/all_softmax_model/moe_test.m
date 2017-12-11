function Y_pred = moe_test(X_test,MOE_model,numofClass,k)
    %p(y|x,theda)
    W = MOE_model.W;
    V = MOE_model.V;
    
    n = size(X_test,1);
    d = size(X_test,2)+1;
    m = numofClass;
    
    X_test_ext = [X_test,ones(n,1)];
    
    pi_i_k = reshape(reshape(exp(X_test_ext*V)./(sum(X_test_ext*V,2)*ones(1,k)),1,[])'*ones(1,m),n,m*k);
    p_n_k = exp(X_test_ext*W)./repmat(reshape(sum(reshape(X_test_ext*W,n*m,[]),2),n,[]),1,k);
    pred = reshape((pi_i_k.*p_n_k)',m,[])';
    [Y_pred_p,Y_pred_i] = max(pred,[],2);
    [~,Y_pred_li] = max(reshape(Y_pred_p,n,[]),[],2);
    Y_pred = Y_pred_i((Y_pred_li-1)*n+(1:n)') - 1;
   
end