dataset=load('data_300_300.mat');
data=dataset.data;
labels=dataset.labels;
clear dataset;
%% train random forests
n=2500;
numoftreeslist=10:10:200;
numoftreeslist=[1 numoftreeslist];
randsamplelist1=randi(n,n,1);
X_old=data(randsamplelist1,:);
Y=labels(randsamplelist1,:);
[~,X,~]=pca(X_old);
for j=1:size(numoftreeslist,2)
    numoftrees=numoftreeslist(j);
    Xtrain=X(1:2000,:);
    Xtest=X(2001:2500,:);
    Ytrain=Y(1:2000,:);
    Ytest=Y(2001:2500,:);
    Ytrainhat=zeros(2000,numoftrees);
    Ytesthat=zeros(500,numoftrees);
    for t=1:numoftrees
        randsamplelist2=randi(2000,2000,1);
        %randfeaturelist=randperm(2499);
        Xtrain2=Xtrain(randsamplelist2,:);
        Ytrain2=Ytrain(randsamplelist2,:);
        tc = fitctree(Xtrain2,Ytrain2);
        for i=1:2000
            Ytrainhat(i,t)=predict(tc,Xtrain(i,:));
        end
        for i=1:500
            Ytesthat(i,t)=predict(tc,Xtest(i,:));
        end
    end

    YtrainRF=sum(Ytrainhat,2)>(numoftrees/2);
    trainCCR=sum(YtrainRF==Ytrain)/2000;
    fprintf('The train CCR wiht forest size = %d is: %f.\n',numoftrees,trainCCR);
    
    YtestRF=sum(Ytesthat,2)>(numoftrees/2);
    testCCR=sum(YtestRF==Ytest)/500;
    fprintf('The test CCR wiht forest size = %d is: %f.\n',numoftrees,testCCR);
end
 
%% Compute performance parameters
for i=1:2000
    Ytrainhat(i,t)=predict(tc,Xtrain(i,:));
end
fprintf('The train CCR is: %f.\n',trainCCR);

for i=1:500
    Ytesthat(i,t)=predict(tc,Xtest(i,:));
end
fprintf('The testCCR is: %f.\n',testCCR);