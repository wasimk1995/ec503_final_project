load('ccr_p_r_f.mat');
figure
% subplot(1,4,1)
hold on
plot(1:2:51,trainCCR)
plot(1:2:51,testCCR)
xlabel('number of trees in the forest')
ylabel('value of CCR')
legend('train CCR','test CCR')
title 'training and test CCR'
hold off

% subplot(1,4,2)
% hold on
% plot(1:2:51,trainP)
% plot(1:2:51,testP)
% xlabel('number of trees in the forest')
% ylabel('value of Precision')
% legend('train Precision','test Precision')
% title 'training and test Precision'
% hold off
% 
% subplot(1,4,3)
% hold on
% plot(1:2:51,trainR)
% plot(1:2:51,testR)
% xlabel('number of trees in the forest')
% ylabel('value of Recall')
% legend('train Recall','test Recall')
% title 'training and test Recall'
% hold off
% 
% subplot(1,4,4)
% hold on
% plot(1:2:51,trainF)
% plot(1:2:51,testF)
% xlabel('number of trees in the forest')
% ylabel('value of F-score')
% legend('train F-score','test F-score')
% title 'training and test F-score'
% hold off