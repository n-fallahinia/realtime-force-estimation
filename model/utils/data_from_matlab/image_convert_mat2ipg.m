clc
clear
close all

subjectIdx = 4;
remove_flag = false;
data_folder = '/home/navid/realtime-force-estimation/dataset_2';
subject_folder = sprintf('%s/subj_%02d', data_folder,subjectIdx);

if (~exist(subject_folder,'dir'))
    error('subjectfolder directory does not exist!')
end

aligned_mat_folder = sprintf('%s/aligned_images', subject_folder);
destination_folder = sprintf('%s/images', subject_folder);

if (~exist(aligned_mat_folder,'dir'))
    error('aligned image directory does not exist!')
end

if (~exist(destination_folder,'dir'))
    fprintf('destination_folder does not exist.... creating one');
    mkdir(destination_folder);   
end

fprintf('images will be saved in %s \n',destination_folder);

image_files = dir(sprintf('%s/aligned_*.mat', aligned_mat_folder));
num_images = length(image_files);

for imageIdx = 1:num_images
    
    if (remove_flag)
        if (mod(imageIdx,50) == 0)
            fprintf('Removing %d images from destination folder\n', imageIdx);
        end
        curent_image = sprintf('%s/image_%04d.jpg', destination_folder, imageIdx);
        delete(curent_image);
    else
        if (mod(imageIdx,50) == 0)
            fprintf('Converted %d images from mat to jpeg\n', imageIdx);
        end
        cuurent_image = image_files(imageIdx).name;
        load(sprintf('%s/%s',aligned_mat_folder,cuurent_image));

        imwrite(imgAligned,sprintf('%s/image_%04d.jpg',destination_folder,imageIdx));
    end    
end
fprintf('\tFinished!');
