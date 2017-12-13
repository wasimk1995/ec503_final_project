function MOE_model = moe_train(X_train,Y_train,k,t_max_em,t_max_grad,lambda,eta)
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
        indicator_y = repmat([Y_train == 0,Y_train == 1],1,k);
        %k Mixture Conditional Priors on x
        %nxk
        %Softmax(V*xi)k
        pi_mixture_old = exp(X_train_ext*V)./(sum(exp(X_train_ext*V),2)*ones(1,k));
        %k Base Posterior Distribution estimates
        %nxm*k
        %Softmax(W_k*xi)
        p_experts_old = exp(X_train_ext*W)./repmat(sum(reshape(exp(X_train_ext*W),n,m,k),3),1,k);
        %nxk
        
        R_ik_temp = pi_mixture_old.*reshape(sum(reshape(p_experts_old.*indicator_y,n,m,k),2),n,k);
        R_ik_old = R_ik_temp./(sum(R_ik_temp,2)*ones(1,k));
        
        R_ik_ind = reshape(repmat(R_ik_old,m,1),n,[]).*indicator_y;
        m_likelihood = sum(sum(X_train_ext*V.*R_ik_old,1)) - sum(sum(R_ik_old.*log(sum(exp(X_train_ext*V),2).*ones(1,k)),2));
        e_likelihood = sum(sum(X_train_ext*W.*R_ik_ind),2) - sum(sum(R_ik_ind.*log(repmat(sum(reshape(exp(X_train_ext*W),n,m,k),3),1,k))));
        likelihood_old = m_likelihood + e_likelihood;
        
        if(isnan(R_ik_old))
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
            eta_t = eta/j;
            R_ik_ind = reshape(repmat(R_ik_old,m,1),n,[]).*indicator_y;
            m_l_l_old = sum(sum(X_train_ext*V.*R_ik_old,1)) - sum(sum(R_ik_old.*log(sum(exp(X_train_ext*V),2).*ones(1,k)),2));
            V = logistic_reg_train_mixture(X_train_ext,Y_train,V,R_ik_old,k,eta_t,lambda);
            m_l_l(j) = sum(sum(X_train_ext*V.*R_ik_old,1)) - sum(sum(R_ik_old.*log(sum(exp(X_train_ext*V),2).*ones(1,k)),2));
            if(m_l_l(j)+10^-5 < m_l_l_old)
                disp("Grad V likelihood decreased");
            end
            m_l_e_old = sum(sum(X_train_ext*W.*R_ik_ind),2) - sum(sum(R_ik_ind.*log(repmat(sum(reshape(exp(X_train_ext*W),n,m,k),3),1,k))));
            W = logistic_reg_train_experts(X_train_ext,Y_train,W,R_ik_old,k,eta_t,lambda);
            m_l_e(j) = sum(sum(X_train_ext*W.*R_ik_ind),2) - sum(sum(R_ik_ind.*log(repmat(sum(reshape(exp(X_train_ext*W),n,m,k),3),1,k))));
            if(m_l_e(j)+10^-5 < m_l_e_old)
                disp("Grad W likelihood decreased")
            end
        end
        
        %k Mixture Conditional Priors on x
        %nxk
        %Softmax(V*xi)k
        pi_mixture_new = exp(X_train_ext*V)./(sum(exp(X_train_ext*V),2)*ones(1,k));
        %k Base Posterior Distribution estimates
        %nxm*k
        %Softmax(W_k*xi)
        p_experts_new = exp(X_train_ext*W)./repmat(sum(reshape(exp(X_train_ext*W),n,m,k),3),1,k);
        %nxk
        
        R_ik_temp = pi_mixture_new.*reshape(sum(reshape(p_experts_new.*indicator_y,n,m,k),2),n,k);
        R_ik_new = R_ik_temp./(sum(R_ik_temp,2)*ones(1,k));
        
        R_ik_ind = reshape(repmat(R_ik_new,m,1),n,[]).*indicator_y;
        m_likelihood = sum(sum(X_train_ext*V.*R_ik_new,1)) - sum(sum(R_ik_new.*log(sum(exp(X_train_ext*V),2).*ones(1,k)),2));
        e_likelihood = sum(sum(X_train_ext*W.*R_ik_ind),2) - sum(sum(R_ik_ind.*log(repmat(sum(reshape(exp(X_train_ext*W),n,m,k),3),1,k))));
        likelihood_new = m_likelihood + e_likelihood;
        like_t(t) = likelihood_new;
        like_diff_t(t) = likelihood_new-likelihood_old;
        
        %disp(["Old",likelihood_old])
        %disp(["New",likelihood_new])
        %Does Likelihood monotonically increase
        if (likelihood_new < likelihood_old)
            disp("Uh-Oh Likelihood decreased")
            break
        elseif (likelihood_new - likelihood_old < 10^-4)
            disp("Likelihood not changing much")
            break
        else
            disp(likelihood_new - likelihood_old)
        end
    end
    MOE_model.like_t = like_t;
    MOE_model.like_diff_t = like_diff_t;
    MOE_model.W = W;
    MOE_model.V = V;
end

function V_out = logistic_reg_train_mixture(X_train,Y_train,V,R,k,eta,lambda)
    n = size(X_train,1);
    d = size(X_train,2);
    m = size(unique(Y_train),2);

    V_x = X_train*V;
    p_y_x_D = exp(V_x)./(sum(exp(V_x),2)*ones(1,k));
    grad_nll = transpose(X_train)*(R.*(p_y_x_D - 1));
    grad_f = grad_nll + lambda*V;
    V_out = V  - eta*grad_f;
    
end

function W_out = logistic_reg_train_experts(X_train,Y_train,W,R,k,eta,lambda)
    n = size(X_train,1);
    d = size(X_train,2);
    m = size(unique(Y_train),1);

    %Use linear indexing into W and V
    indicator_y = reshape([(repmat(Y_train,1,k) == 0);(repmat(Y_train,1,k) == 1)],n,m*k);
    
    W_x = X_train*W;
    p_y_x_D = exp(W_x)./repmat(sum(reshape(exp(W_x),n,m,k),3),1,k);
    R_imk = reshape(repmat(R,m,1),n,[]);
    indicator_R = R_imk.*indicator_y;
    grad_nll = transpose(X_train)*(indicator_R.*(p_y_x_D - indicator_y));
    grad_f = grad_nll + lambda*W;
    W_out = W - eta*grad_f;
end