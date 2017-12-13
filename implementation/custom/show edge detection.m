label_dir = 'D:/graduate/ec503/project/face recognition/VOCdevkit/VOC2012/Annotations/';
image_dir = 'D:/graduate/ec503/project/face recognition/VOCdevkit/VOC2012/JPEGImages/';
xml_listing = dir(label_dir);
jpeg_listing = dir(image_dir);
x_n = size(xml_listing,1);
pixels = [300,300];
dimensions = pixels(1)*pixels(2);
labels = zeros(x_n-2,1);
data = zeros(x_n-2,dimensions);
for i=3:3%x_n
    %EXTRACT FEATURES
    color_image = imread([image_dir,jpeg_listing(i).name]);
    grey_image = rgb2gray(imresize(color_image,pixels));
    edges = edge(grey_image,'Sobel');
    data(i-2,:) = edges(:)';
    figure
    subplot(1,3,1)
    imshow(color_image)
    title('rgb picture')
    subplot(1,3,2)
    imshow(grey_image)
    title('greyscale picture')
    subplot(1,3,3)
    imshow(edges)
    title('edge picture')
end