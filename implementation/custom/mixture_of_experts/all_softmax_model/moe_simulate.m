%LOAD DATA
%load('data_300_300.mat');
disp("Data Loaded")

%%
%CREATE SMALLER SET AND PARTITION DATA
%Work on smaller data set
p_data = .01;
size_data = size(data,1);
data_i = false(size_data,1);
data_i(1:round(p_data*size_data)) = true;
X = data(data_i,:);
Y = labels(data_i,:);
[~,X,~] = pca(X);

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
disp("Data Partitioned")

%%
%TRAIN DATA
k = 10;
MOE_model = moe_train(X_train,Y_train,k);

%%
%TEST DATA
Y_pred = moe_test(X_test,MOE_model,size(unique(Y_train),1),k);

%%
CCR = sum(Y_pred == Y_test)/size(Y_test,1)