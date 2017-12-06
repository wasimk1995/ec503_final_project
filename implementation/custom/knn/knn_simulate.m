%LOAD DATA
load('data_300_300.mat');

%%
%CREATE SMALLER SET AND PARTITION DATA
%Work on smaller data set
p_data = 1;
size_data = size(data,1);
data_i = false(size_data,1);
data_i(1:round(p_data*size_data)) = true;
X = data(data_i,:);
Y = labels(data_i,:);

%Divide data into training and testing
p_train = .7;
n_x = size(X,1);
train_i = false(n_x,1);
train_i(1:round(p_train*n_x)) = true;
train_i = train_i(randperm(n_x));
X_train = X(train_i,:);
Y_train = Y(train_i,:);
X_test = X(~train_i,:);
Y_test = Y(~train_i,:);

%%
%TRAIN DATA
knn_model = knn_train(X_train,Y_train);

%%
%TEST DATA
k = 15;
for i=1:k
    [Y_pred,posteriors] = knn_test(X_test,knn_model,i);
    tmper = [i,sum(Y_pred == Y_test) / size(Y_test,1)]
    accuracy(i) = sum(Y_pred == Y_test) / size(Y_test,1);
end

%%
x=1;