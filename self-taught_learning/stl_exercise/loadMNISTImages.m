function [images labels] = loadMNISTImages(type)

% Load MNIST images from mnist_all.mat
% Use this mat to replace original MNIST dataset

load('mnist_all.mat');

images = [];
labels = [];
if strcmp(type, 'train')
    for i= 0:9
        temp_name = strcat(type, num2str(i));
        temp = eval(temp_name);
        labels = [labels; ones(size(temp,1), 1) * i];
        images = [images temp'];
        images = double(images);
    end
else if strcmp(type, 'test')
        for i = 0:9
            temp_name = strcat(type, num2str(i));
            temp = eval(temp_name);
            labels = [labels; ones(size(temp,1), 1) * i];
            images = [images temp'];
            images = double(images);
        end
    end    
end


