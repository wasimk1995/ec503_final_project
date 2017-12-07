% dataset=load('data_300_300.mat');
% X=dataset.data;
% Y=dataset.labels;
% clear dataset
%% train random forests
n=17125;
numofclass=2;
numoftrees=10;
D=90000;
m=150;
CART=zeros(m,numoftrees);
for t=1:numoftrees
    randsamplelist=randi(n,n,1);    randfeaturelist=randperm(D);
    randfeaturelist=randfeaturelist(1,1:1000);
    for j=1:m
        idx1=Y(randsamplelist,1)==1;
        idx0=Y(randsamplelist,1)==0;
        n1=sum(idx1);
        n0=sum(idx0);
        p1=n1/(n1+n0);
        p0=n0/(n1+n0);
        nf=size(randfeaturelist,2);
        Gini=zeros(numofclass,nf);
        for i=1:nf
            idx=randfeaturelist(i);
            pa=sum(X(idx1,idx)==1)/n1;
            pb=sum(X(idx1,idx)==0)/n1;
            pc=sum(X(idx0,idx)==1)/n0;
            pd=sum(X(idx0,idx)==0)/n0;
            gini1=1-pa*pa-pb*pb;
            gini2=1-pc*pc-pd*pd;
            Gini(1,i)=p1*gini1+p0*gini2;
            Gini(2,i)=p0*gini2+p1*gini1; % because there are only two classes, they are the same. 
        end
        gini=Gini(1,:)';
        [minval,minidx]=min(gini);
        idx=randfeaturelist(minidx);
        CART(j,t)=idx;
        randfeaturelist(minidx)=[];
        A=find(X(:,idx)==1);
        randsamplelist=randsamplelist(~ismember(randsamplelist,A));
    end
end

%% test
Ypredicted=zeros(n,numoftrees);
for t=1:numoftrees
    for i=1:n
        if sum(X(i,CART(:,t)))~=0
            Ypredicted(i,t)=1;
        end
    end
end
y=sum(Ypredicted,2)>=(numoftrees/2);
CCR=sum(y==Y)/n;
fprintf('The CCR with numoftrees=%d m=%d is: %f.\n',numoftrees,m,CCR);