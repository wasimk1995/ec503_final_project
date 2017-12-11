%LOAD DATA
%load('data_300_300.mat');
disp("Data Loaded")

%%
% %CREATE SMALLER SET AND PARTITION DATA
% %Work on smaller data set
% p_data = .01;
% size_data = size(data,1);
% data_i = false(size_data,1);
% data_i(1:round(p_data*size_data)) = true;
% X = data(data_i,:);
% Y = labels(data_i,:);

%%
%Create Synthetic Data
X=[1:1000,2001:3000,4001:5000,6001:7000,8001:11000];
X = (X/size(X,2))';

Y(1:1000,1) = (.2*(1:1000)+100+.05*rand(1,1000))';
Y(1001:2000,1) = (-.4*(1001:2000)+50+.04*rand(1,1000))';
Y(2001:3000,1) = (.5*(2001:3000)+200+.01*rand(1,1000))';
Y(3001:4000,1) = (-.1*(3001:4000)+832+.03*rand(1,1000))';
Y(4001:7000,1) = (-.1*(4001:7000)+832+.12*rand(1,3000))';
Y = Y/sum(Y);
%scatter(X,Y,1)

%Divide data into training and testing
p_train = .7;
n = size(X,1);
train_i = false(n,1);
train_i(1:round(p_train*n)) = true;
train_i = train_i(randperm(n));
X_train = X(train_i,:);
Y_train = Y(train_i,:);
X_test = X(~train_i,:);
Y_test = Y(~train_i,:);
disp("Data Partitioned")
%figure, scatter(X_train,Y_train);
%figure, scatter(X_test,Y_test);

%for k=1:5
%%
%TRAIN DATA
k=5;
t_max_em = 300;
t_max_grad = 500;
lambda = 1;
MOE_model = moe_train(X_train,Y_train,k,t_max_em,t_max_grad,lambda);

%%
%TEST DATA
Y_pred = moe_test(X_test,Y_test,MOE_model,k);
figure, scatter(X_test,Y_test,1,'b')
hold on
scatter(X_test,Y_pred,1,'r')
title('Mixture of Experts k=5')
xlabel('x')
ylabel('y')
x=1;
%end