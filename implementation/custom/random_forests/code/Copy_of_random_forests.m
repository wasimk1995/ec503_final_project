dataset=load('data_300_300.mat');
X=dataset.data;
Y=dataset.labels;
clear dataset

%% train random forests
n=17125;
numofclass=2;
numoftrees=100;
D=90000;
CART=zeros(10,100);
for t=1:1%100
    randsamplelist=randi(17125,17125,1);
    idx1=Y(randsamplelist,1)==1;
    idx0=Y(randsamplelist,1)==0;
    n1=sum(idx1);
    n0=sum(idx0);
    p1=n1/n;
    p0=n0/n;
    randfeaturelist=randperm(D);
    randfeaturelist=randfeaturelist(1,1:1000);
    Gini=zeros(numofclass,1000);
    for j=1:10
        for i=1:size(randfeaturelist,2)        
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
    end
    
end