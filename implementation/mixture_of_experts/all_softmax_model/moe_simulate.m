%LOAD DATA
%load('data_300_300.mat');
disp("Data Loaded")

%%
% %CREATE SMALLER SET AND PARTITION DATA
% %Work on smaller data set
% p_data = .1;
% size_data = size(data,1);
% data_i = false(size_data,1);
% data_i(1:round(p_data*size_data)) = true;
% X = data(data_i,:);
% Y = labels(data_i,:);
% [~,X,~] = pca(X);
% % 

%%
% %Create Synthetic Data
X(1:100,1:2) = ([1:100;(-1*(1:100))+(40.5*rand(1,100))])';
X(101:200,1:2) = ([101:200;(.5*(1:100))+(25*rand(1,100))])';
X(201:300,1:2) = ([201:300;(-.25*(1:100))+(20*rand(1,100))])';
X(301:400,1:2) = ([301:400;(.3*(1:100))+(25*rand(1,100))])';
X(401:500,1:2) = ([401:500;(1*(1:100))+(30*rand(1,100))])';

X = 1000*X./(ones(size(X,1),1)*sum(X,1));

Y(1:100,1) = 0;
Y(101:200,1) = 1;
Y(201:300,1) = 0;
Y(301:400,1) = 1;
Y(401:500,1) = 0;

figure,gscatter(X(:,1),X(:,2),Y,'rb');
xlabel('x')
ylabel('y')
legend('Class 1','Class 2')
title('True Labels')

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

figure,gscatter(X_test(:,1),X_test(:,2),Y_test,'rb');
xlabel('x')
ylabel('y')
legend('Class 1','Class 2')
title('True Labels')

for k=2:10
    %%
    %TRAIN DATA
    t_max_em=600;
    t_max_grad=600;
    lambda = 1;
    eta = 10^-5;
    MOE_model = moe_train(X_train,Y_train,2^k,t_max_em,t_max_grad,lambda,eta);

    %%
    %TEST DATA
    Y_pred = moe_test(X_test,MOE_model,size(unique(Y_train),1),2^k);

    %%
    CCR(k) = sum(Y_pred == Y_test)/size(Y_test,1)
    % likelihood(k,:) = MOE_model.like_t;
    % likehood_diff(k,:) = MOE_model.like_diff_t;
    % plot(1:size(MOE_model.like_t,2),MOE_model.like_t);
    % title("Expected Log Likelihood vs EM iterations k=2 n=1000 d=2")
    % xlabel('t');
    % ylabel('ELL');
    % figure,plot(1:size(MOE_model.like_diff_t,2),MOE_model.like_diff_t);
    % xlabel('t');
    % ylabel('ELL(t) - ELL(t-1)');
    % title("ELL(t) - ELL(t-1) vs EM iterations k=2 n=1000 d=2")
    figure,gscatter(X_test(:,1),X_test(:,2),Y_pred,'rb');
    xlabel('x')
    ylabel('y')
    title(['Predicted Labels k=',char(48+k)]);
    legend('Class 1','Class 2')
end