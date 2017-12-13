function Y_pred = moe_test(X_test,Y_test,MOE_model,k)
    %p(y|x,theda)
    Sigma = MOE_model.Sigma;
    W = MOE_model.W;
    V = MOE_model.V;
    n = size(X_test,1);
    
    X_test_ext = [X_test,ones(n,1)];
     
    pi_mixture = exp(X_test_ext*V)./(sum(exp(X_test_ext*V),2)*ones(1,k));
    pi_experts = X_test_ext*W;
    Y_pred = sum(pi_mixture.*pi_experts,2);
    
end