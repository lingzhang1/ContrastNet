% read data
mainpath = './data/modelnet40_ply_hdf5_2048_cut8';

data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

for n=1:length(data_files)
    data_path = strcat( mainpath, '/',data_files(n).name);

%     h5disp(data_path);
    info = h5info(data_path);
    cut1 = info.Datasets(1).Name;
    if strcmp(cut1,'cut1') == 0
       data_path
       delete(data_path);
    end   
end
