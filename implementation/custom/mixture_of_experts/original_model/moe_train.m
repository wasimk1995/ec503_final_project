function MOE_model = moe_train(X_train,Y_train,k,t_max_em,t_max_grad,lambda)
    n = size(X_train,1);
    d = size(X_train,2) + 1;
    m = unique(Y_train);
    X_train_ext = [X_train,ones(n,1)];
    
    %Initialize parameters
    %k set of mixture weights
    %dxk
    V = rand(d,k)/d;   
    %k set of base distribution weights for Gaussian mean prediction
    %dxk
    W = rand(d,k)/d;    
    %k sigmas for Gaussian base distributions
    %1xk
    Sigma = rand(k,1);                                                  
    
    for t=1:t_max_em
        %k Mixture Conditional Priors on x
        %nxk
        V_x_old = X_train_ext*V;
        pi_mixture_old = (exp(V_x_old)./(sum(exp(V_x_old),2)*ones(1,k)));
        %k Base Posterior Distribution estimates
        %nxm*k
        p_experts_old = exp(-((Y_train*ones(1,k)-X_train_ext*W).^2)/2./(ones(n,1)*Sigma'))./sqrt(2*pi*ones(n,1)*(Sigma'));                
        R_ik_old_temp = pi_mixture_old.*p_experts_old; 
        R_ik_old_temp(R_ik_old_temp < 10^-10) = 10^-10;
        R_ik_old = R_ik_old_temp./(sum(R_ik_old_temp,2)*ones(1,k));
        
        %Debugging
        %At each iteration, check if the likelihood has increased. Make
        %sure likelihood never decreases
        %p(y|x,theda)
        temp_tot = pi_mixture_old.*p_experts_old;
        temp_tot(temp_tot < 10^-10) = 10^-10;
        likelihood_old = sum(sum(R_ik_old.*log(temp_tot),2));
        
        %ESTIMATE NEW PARAMETERS
        %Mixture Weights
        %Modeled by Multinomial Logistic Regressions, Softmax
        %Use Gradient Descent
        %v_d_k
        V_old = V;
        for j=1:t_max_grad
            eta=10^-5;
            grad_l_old = sum(sum(X_train_ext*V.*R_ik_old,1)) - sum(sum(R_ik_old.*log(sum(exp(X_train_ext*V),2).*ones(1,k)),2));
            V = logistic_reg_train_mixture(X_train_ext,Y_train,V,R_ik_old,k,eta,lambda);
            grad_likelihood(j) = sum(sum(X_train_ext*V.*R_ik_old,1)) - sum(sum(R_ik_old.*log(sum(exp(X_train_ext*V),2).*ones(1,k)),2));
            if(grad_likelihood(j) - grad_l_old < 10^-15)
                break
            end
        end
        
        %disp(W);
        %Parameters of Mixture Densities
        %Modeled by Binary Logistic Regression, Weighted Least Squares
        %sigma_k,w_d_k
        W_old = W;
        for l=1:k
            W(:,l) = inv(X_train_ext'*diag(R_ik_old(:,l))*X_train_ext)*X_train_ext'*diag(R_ik_old(:,l))*Y_train;
            Sigma(l) = sum(R_ik_old(:,l).*((Y_train-X_train_ext*W(:,l)).^2))/sum(R_ik_old(:,l));
        end
 
        
        %Set new parameters
        V_x_new = X_train_ext*V;
        pi_mixture_new = (exp(V_x_new)./(sum(exp(V_x_new),2)*ones(1,k)));
        p_experts_new = exp(-((Y_train*ones(1,k)-X_train_ext*W).^2)/2./(ones(n,1)*Sigma'))./sqrt(2*pi*ones(n,1)*(Sigma'));                
        R_ik_new_temp = pi_mixture_new.*p_experts_new; 
        R_ik_new_temp(R_ik_new_temp < 10^-10) = 10^-10;
        R_ik_new = R_ik_new_temp./(sum(R_ik_new_temp,2)*ones(1,k));
        
        temp_tot = pi_mixture_new.*p_experts_new;
        temp_tot(temp_tot < 10^-20) = 10^-20;
        likelihood_new = sum(sum(R_ik_new.*log(temp_tot),2));
        %disp(["Old",likelihood_old])
        %disp(["New",likelihood_new])
        like_plot(t) = likelihood_new;
        diff_plot(t) = (likelihood_new - likelihood_old);
        if(ismissing(likelihood_new))
            break;
        end
        %Does Likelihood monotonically increase
        if (likelihood_new+10^-5 < likelihood_old)
            disp("Uh-Oh Likelihood decreased")
            break
        elseif (likelihood_new - likelihood_old < 10^-5)
            disp("Likelihood not changing much")
            break
        end
    end
    
    MOE_model.Sigma = Sigma;
    MOE_model.W = W;
    MOE_model.V = V;
end

function W_out = logistic_reg_train_mixture(X_train,Y_train,W,R,k,eta,lambda)
    n = size(X_train,1);
    d = size(X_train,2);
    m = size(unique(Y_train),2);

    W_x = X_train*W;
    p_y_x_D = exp(W_x)./(sum(exp(W_x),2)*ones(1,k));
    grad_nll = transpose(X_train)*((R.*p_y_x_D - R));
    grad_f = grad_nll + lambda*W;
    W_out = W  - eta*grad_f;
    if(sum(sum(isnan(W_out))))
       coo = 1 ;
    end
end