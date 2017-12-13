%% generate data and labels
rand('state', 0);
randn('state', 0);
N= 50;
D= 2;

X1 = mgd(N, D, [4 3], [2 -1;-1 2]);
X2 = mgd(N, D, [1 1], [2 1;1 1]);
X3 = mgd(N, D, [3 -3], [1 0;0 4]);

X= [X1; X2; X3];
X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];

%% plot
figure
hold on
plot(X(Y==1,1), X(Y==1,2), 'o', 'MarkerFaceColor', [.9 .3 .3], 'MarkerEdgeColor','k');
plot(X(Y==2,1), X(Y==2,2), 'o', 'MarkerFaceColor', [.3 .9 .3], 'MarkerEdgeColor','k');
plot(X(Y==3,1), X(Y==3,2), 'o', 'MarkerFaceColor', [.3 .3 .9], 'MarkerEdgeColor','k');
x1=[-1.5,1.5];
y1=-0.207485+0*x1;
plot(x1,y1,'k')
y2=[-0.207485,0.8];
x2=0.327461+0*y2;
plot(x2,y2,'k')
x3=[-1.5,0.327461];
y3=0.302167+0*x3;
plot(x3,y3,'k')
y4=[-0.207485,0.302167];
x4=-0.28321+0*y4;
plot(x4,y4,'k')
x5=[-0.28321,0.327461];
y5=-0.0280978+0*x5;
plot(x5,y5,'k')
tc = fitctree(X,Y);
view(tc,'mode','graph')
