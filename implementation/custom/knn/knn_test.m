function [Y_predict,Posteriors] = knn_test(X_test,KNNmodel,k)
    disp("Entered knn_test")
    X_train = KNNmodel.X_train;
    Y_train = KNNmodel.Y_train;
    
    Y_unique = unique(Y_train);
    n_x_test = size(X_test,1);
    numofClass=size(Y_unique,1);
    
    preprocess_train = KNNmodel.preprocess_train;
    
    %Find X_test*X_train' and test
    distances = -2*(X_test*transpose(X_train)) + ones(n_x_test,1)*preprocess_train;
    [dist_class_sorted(:,:,1),I] = sort(distances,2);
    dist_class_sorted(:,:,2) = Y_train(I);
    closest_to_farthest_classes = dist_class_sorted(:,:,2);

    %Run the rule by finding the max sum over k for each class and get
    %posterior
    for k_i=1:k
        for i=1:numofClass
            class_matches = (closest_to_farthest_classes == Y_unique(i));
            sum_classes(:,i) = sum(class_matches(:,1:k_i),2);
            Posteriors(:,i) = sum_classes(:,i)/k_i;
        end
    
        [~,Y_predict_i(:,k_i)] = max(sum_classes,[],2);
    end
    Y_predict = Y_unique(Y_predict_i);
    disp("Exiting knn_test")
end