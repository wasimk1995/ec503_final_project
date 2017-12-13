load('original result.mat')
figure
plot(10:10:190,result_different_m)
xlabel('the number of nodes (m)')
ylabel('the value of CCR')
title('plot of CCR with different amount of nodes')

figure
plot(10:10:100,result_different_treenum)
xlabel('the number of trees')
ylabel('the value of CCR')
title('plot of CCR with different amount of trees in forest')