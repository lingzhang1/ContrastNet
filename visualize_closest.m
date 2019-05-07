% read data
mainpath = 'data/shapenet_cut';

data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

f = fopen('data/shapenet_cut/train_files.txt');
data_files = textscan(f,'%s');
fclose(f);
data_files = data_files{1}(:);

f = fopen('closed_index.txt');

formatSpec = '%d %d %d %d %d %d %d %d %d %d %d ';
sizeA = [11 Inf];

A = fscanf(f,formatSpec,sizeA);
fclose(f);
A = A';
files = A;


% for n=1:length(files(:,1))
    for m=1:length(files(n,:))
        data_path = data_files{files(6,m)+1,:};
        cut1 = h5read(data_path,'/cut1');
        cut1 = cut1';
        cut2 = h5read(data_path,'/cut2');
        cut2 = cut2';
        xyzPoints = cat(1,cut1,cut2);
        
        figure;
        pcshow(xyzPoints, 'MarkerSize', 150);
    end
% end

