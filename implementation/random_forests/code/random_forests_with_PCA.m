dataset=load('data_300_300.mat');
data=dataset.data;
labels=dataset.labels;
clear dataset;
%% train and test random forests
n=2500;
numoftreeslist=1:2:51;
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
    A=sum(YtrainRF.*Ytrain);
    P=A/sum(YtrainRF);
    R=A/sum(Ytrain);
    F=2*P*R/(P+R);
    trainCCR=sum(YtrainRF==Ytrain)/2000;
    fprintf('When forest size = %d, trainCCR, Precision, Recall, F-score are: %f,%f,%f,%f.\n',numoftrees,trainCCR,P,R,F);
    
    YtestRF=sum(Ytesthat,2)>(numoftrees/2);
    A=sum(YtestRF.*Ytest);
    P=A/sum(YtestRF);
    R=A/sum(Ytest);
    F=2*P*R/(P+R);
    testCCR=sum(YtestRF==Ytest)/500;
    fprintf('When forest size = %d, testCCR, Precision, Recall, F-score are: %f,%f,%f,%f.\n',numoftrees,testCCR,P,R,F);
    
end
 
