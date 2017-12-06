function [KNNmodel] = knn_train(X_train, Y_train)
    KNNmodel.X_train = X_train;
    KNNmodel.Y_train = Y_train;
    
    iterations = 4;
    n_x_train = size(X_train,1);
    n_x_train_i=round(n_x_train/iterations);
    %Caclulate all x*x' for training set
    for i=1:iterations
        start_index = (i-1)*n_x_train_i+1;
        end_index = n_x_train_i*i;
        if i==iterations
            end_index = n_x_train;
        end
        preprocess_train(1,start_index:end_index) = diag(X_train(start_index:end_index,:)*transpose(X_train(start_index:end_index,:)));
    end
    
    KNNmodel.preprocess_train = preprocess_train;
    
end