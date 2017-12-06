I = imread('/Users/wasimkhan/Desktop/BostonUniversity/grad/learning_from_data/project/datasets/VOC2012/JPEGImages/2011_005847.jpg');
I = rgb2gray(I);
BW1 = edge(I,'Canny');
imshow(BW1);