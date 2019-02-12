mainpath = './data/modelnet40_ply_hdf5_2048_cut';
data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

train_out = './data/modelnet40_ply_hdf5_2048_cut/trian_files.txt';
train_fileID = fopen(train_out,'w');

test_out = './data/modelnet40_ply_hdf5_2048_cut/test_files.txt';
test_fileID = fopen(test_out,'w');

for n=1:length(data_files)
    data_path = strcat( mainpath, '/',data_files(n).name); 
    
    trian = strfind(data_path,'trian');
    if isempty(trian) == 0
        fprintf(train_fileID, '%s\n', data_path);
    end
    
    test = strfind(data_path,'test');
    if isempty(test) == 0
        fprintf(test_fileID, '%s\n', data_path);
    end
end
fclose(train_fileID);
fclose(test_fileID);

