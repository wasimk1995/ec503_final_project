function MOE_model = moe_train(X_train,Y_train,k)
    t_max_em=100;
    t_max_grad=500;
    lambda = 1;
    n = size(X_train,1);
    d = size(X_train,2) + 1;
    m = size(unique(Y_train),1);
    X_train_ext = [X_train,ones(n,1)];
    
    %Initialize parameters
    %k set of mixture weights
    %dxk
    V = rand(d,k);   
    %k set of base parameters for base distributions
    %dxk
    W = rand(d,k*m);                                                    
    
    for t=1:t_max_em
        %k Mixture Conditional Priors on x
        %nxk
        %Softmax(V*xi)k
        pi_mixture = exp(X_train_ext*V)./(sum(exp(X_train_ext*V),2)*ones(1,k));
        %k Base Posterior Distribution estimates
        %nxm*k
        %Softmax(W_k*xi)
        p_experts = exp(X_train_ext*W)./reshape(sum(reshape(exp(X_train_ext*W),n*m,k),2)*ones(1,k),n,m*k);
        %nxk
        pi_mix_y = reshape(repmat(reshape(pi_mixture,[],1)',m,1)',n,[]);
        
        indicator_y =reshape(reshape(reshape([(repmat(Y_train,k,1) == 0);(repmat(Y_train,k,1) == 1)],n,k*m),n*m,k),n,m*k);
        R_num = reshape(sum(reshape(indicator_y.*exp(X_train_ext*W),n,m,k),2),n,k);
        R_ik = R_num./(sum(R_num,2)*ones(1,k));
        
        likelihood_old = sum(sum(R_ik,2));
        
        if(isnan(R_ik))
            boo=1;
        end
        
        %ESTIMATE NEW PARAMETERS
        
        %Mixture Weights
        %Modeled by Multinomial Logistic Regressions, Softmax
        %V - d*k
        %Parameters of Mixture Densities
        %K Multinomial Logistic Regression Models
        %W - dxk*m
        for j=1:t_max_grad
            if(j<100)
                eta = 10^-2;
            elseif(j > 400)
                eta = 10^-10;
            else
                eta = 1/j;
            end
            V = logistic_reg_train_mixture(X_train_ext,Y_train+1,V,R_ik,k,eta,lambda);
            W = logistic_reg_train_experts(X_train_ext,Y_train+1,W,R_ik,k,eta,lambda);
        end
        
        %k Mixture Conditional Priors on x
        %nxk
        %Softmax(V*xi)k
        pi_mixture = exp(X_train_ext*V)./(sum(exp(X_train_ext*V),2)*ones(1,k));
        %k Base Posterior Distribution estimates
        %nxm*k
        %Softmax(W_k*xi)
        p_experts = exp(X_train_ext*W)./reshape(sum(reshape(exp(X_train_ext*W),n*m,k),2)*ones(1,k),n,m*k);
        %nxk
        pi_mix_y = reshape(repmat(reshape(pi_mixture,[],1)',m,1)',n,[]);
        
        indicator_y =reshape(reshape(reshape([(repmat(Y_train,k,1) == 0);(repmat(Y_train,k,1) == 1)],n,k*m),n*m,k),n,m*k);
        R_num = reshape(sum(reshape(indicator_y.*exp(X_train_ext*W),n,m,k),2),n,k);
        R_ik = R_num./(sum(R_num,2)*ones(1,k));
        
        likelihood_new = sum(sum(R_ik,2));
        disp(likelihood_new)
        %Does Likelihood monotonically increase
        if (likelihood_new < likelihood_old)
            disp("Uh-Oh Likelihood decreased")
        end
    end
    
    MOE_model.W = W;
    MOE_model.V = V;
end

function W_out = logistic_reg_train_mixture(X_train,Y_train,W,R,k,eta,lambda)
    n = size(X_train,1);
    d = size(X_train,2);
    m = size(unique(Y_train),2);

    W_x = X_train*W;
    p_y_x_D = exp(W_x)./(sum(exp(W_x),2)*ones(1,k));
    grad_nll = transpose(X_train)*(R.*(p_y_x_D - 1));
    grad_f = grad_nll + lambda*W;
    W_out = W - eta*grad_f;
    W_max = max(W_x)
    W_out_max = max(W_out)
    if(sum(sum(isnan(W_out))))
       coo = 1 ;
    end
end

function W_out = logistic_reg_train_experts(X_train,Y_train,W,R,k,eta,lambda)
    n = size(X_train,1);
    d = size(X_train,2);
    m = size(unique(Y_train),1);

    indicator_y = [(repmat(Y_train,k,1) == 0);(repmat(Y_train,k,1) == 1)];
    
    W_x = X_train*W;
    W_max = max(W_x)
    p_y_x_D = exp(W_x)./repmat(reshape(sum(reshape(W_x,n*m,[]),2),n,[]),1,k);
    indicator_R = reshape(indicator_y',n,[]).*reshape(repmat(reshape(R,n*k,1),1,m),n,[]);
    grad_nll = transpose(X_train)*(indicator_R.*(p_y_x_D - 1));
    grad_f = grad_nll + lambda*W;
    W_out = W - eta*grad_f;
    W_out_max = max(W_out)
    if(sum(sum(isnan(W_out))))
       coo = 1 ;
    end
end