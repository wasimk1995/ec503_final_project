function [Y_predict,Posteriors] = knn_test(X_test,KNNmodel,k)
    X_train = KNNmodel.X_train;
    Y_train = KNNmodel.Y_train;
    
    Y_unique = unique(Y_train);
    n_x_test = size(X_test,1);
    numofClass=size(Y_unique);
    
    preprocess_train = KNNmodel.preprocess_train;
    
    %Find X_test*X_train' and test
    distances = -2*(X_test*transpose(X_train)) + ones(n_x_test,1)*preprocess_train;
    [dist_class_sorted(:,:,1),I] = sort(distances,2);
    dist_class_sorted(:,:,2) = Y_train(I);

    %Run the rule by finding the max sum over k for each class and get
    %posterior
    for i=1:numofClass
        class_matches = (dist_class_sorted(:,:,2) == Y_unique(i));
        sum_classes(:,i) = sum(class_matches(:,1:k),2);
        Posteriors(:,i) = sum_classes(:,i)/k;
    end
    
    [~,Y_predict_i] = max(sum_classes,[],2);
    Y_predict = Y_unique(Y_predict_i);
end