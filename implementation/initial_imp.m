label_dir = '../../datasets/VOC2012/Annotations/';
image_dir = '../../datasets/VOC2012/JPEGImages/'
xml_listing = dir(label_dir);
jpeg_listing = dir(image_dir);
x_n = size(xml_listing,1);
pixels = [300,300];
dimensions = pixels(1)*pixels(2);
labels = zeros(x_n-2,1);
data = zeros(x_n-2,dimensions);
for i=3:x_n
    %GET LABELS
    DOMnode = xmlread([label_dir,xml_listing(i).name]);
    objects = DOMnode.getElementsByTagName('object');
    for k=0:objects.getLength-1
        this_object = objects.item(k);
        my_name = this_object.getElementsByTagName('name');
        object_name = my_name.item(0);
        if strcmp(object_name.getFirstChild.getData,'person')
            labels(i-2) = 1;
        end
    end
    %EXTRACT FEATURES
    color_image = imread([image_dir,jpeg_listing(i).name]);
    grey_image = rgb2gray(imresize(color_image,pixels));
    edges = edge(grey_image,'Sobel');
    data(i-2,:) = edges(:)';
end
save('data_300_300.mat','-v7.3','data','labels');