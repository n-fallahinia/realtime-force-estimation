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

force_folder = sprintf('%s/forces', subject_folder);

if (~exist(force_folder,'dir'))
    error('force folder does not exist!')
end

fprintf('forces will be saved in %s \n',force_folder);

force_file = sprintf('%s/params_%02d.mat', force_folder,subjectIdx);
load(force_file)
num_forces = length(force_matrix);

fileID = fopen(sprintf('%s/force_%02d.txt', force_folder,subjectIdx),'w');

for forceIdx = 1:num_forces
    
    if (remove_flag)
%         if (mod(forceIdx,50) == 0)
%             fprintf('Removing %d forces from destination folder\n', forceIdx);
%         end
%         curent_image = sprintf('%s/image_%04d.jpg', destination_folder, imageIdx);
%         delete(curent_image);
    else
        if (mod(forceIdx,50) == 0)
            fprintf('Converted %d forces from mat to txt\n', forceIdx);
        end
        curent_forces = force_matrix(forceIdx,:);
        force_to_write = sprintf('%6.4f,%6.4f,%6.4f\n',curent_forces(1),curent_forces(2),curent_forces(3));
        fprintf(fileID,force_to_write);
    end    
end
fclose(fileID);
fprintf('\tFinished!');