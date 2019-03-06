% read data
mainpath = 'hdf5_data';

data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

%     label_path = strcat( mainpath, '/', category, '/points_label', '/*.seg');
%     label_files = dir(label_path);

mkdir hdf5_data coords_normal;

for n=1:length(data_files)
    data_path = strcat( mainpath, '/', data_files(n).name);
%     xyzPoints = load(data_path);
    
    data = h5read(data_path,'/data');
    label = h5read(data_path,'/label');
    pid = h5read(data_path,'/pid');
    
%     h5disp(data_path);
    
    x = length(data(:, 1, 1)) + 3;
    y = length(data(1, :, 1));
    z = length(data(1, 1, :));
    result = zeros(x,y,z);
    nan_num = 0;
    for i = 1:z
        xyzPoints = data(:,:,i);
        xyzPoints = xyzPoints(1:3,:);
        xyzPoints = xyzPoints';

%%%%%%%%%%%%%%%%  reduction based on label  %%%%%%%%%%%%%%%  

    %         label_path = strcat( mainpath, '/', category, '/points_label/', label_files(n).name);
    %         lables = load(label_path);
    %         A = (lables ~= 2);
    
%%%%%%%%%%%%%%%%  cut part of the object bansed on axis  %%%%%%%%%%%%%%%  
  
    %         [Max_v,Max_i] = max(xyzPoints);
    %         [Min_v,Min_i] = min(xyzPoints);
    %         Range_value = Max_v - Min_v;
    %         [value, axis] = max(Range_value);
    %         A = xyzPoints(:,axis) > (Min_v(axis) + value * 0.3);
    % 
    %         xyzPoints = xyzPoints(A ~= 0,:);

%%%%%%%%%%%%%%%%  get normals  %%%%%%%%%%%%%%%    
        ptCloud = pointCloud(xyzPoints);
        normals = pcnormals(ptCloud, 20);
        
%                 
%         [row, col] = find(isnan(normals));
%         if length(row) ~= 0
%             xyzPoints(row,:) = xyzPoints(row-1,:);
%             normals(row,:) = normals(row-1,:);
%         end
        [row, col] = find(isnan(normals));
        nan_num = nan_num + length(row);

%%%%%%%%%%%%%%%%  show normals  %%%%%%%%%%%%%%%
%         figure;
%         pcshow(ptCloud);
%         title('Estimated Normals of Point Cloud');
%         hold on;
%     
%         x = ptCloud.Location(1:1:end,1);
%         y = ptCloud.Location(1:1:end,2);
%         z = ptCloud.Location(1:1:end,3);
%         u = normals(1:1:end,1);
%         v = normals(1:1:end,2);
%         w = normals(1:1:end,3);
%     
%         quiver3(x,y,z,u,v,w);
%         hold off
   
%%%%%%%%%%%%%%%%  out put normals and coords  %%%%%%%%%%%%%%%
        coords_normal = cat(2, xyzPoints, normals);
        coords_normal = coords_normal';
        result(:,:,i) = coords_normal;
    end
    
    processing = data_files(n).name
    nan_num
    
    out_path = strcat( mainpath, '/coords_normal/', data_files(n).name);
    
    info = h5info(data_path);

%     Dataspace_data = info.Datasets(1).Dataspace.Size;
    ChunkSize_data = info.Datasets(1).ChunkSize;
    
    Dataspace_label = info.Datasets(2).Dataspace.Size;
    ChunkSize_label = info.Datasets(2).ChunkSize;
    
    Dataspace_pid = info.Datasets(3).Dataspace.Size;
    ChunkSize_pid = info.Datasets(3).ChunkSize;
    
    h5create(out_path,'/data',[x y z],'Datatype','single','ChunkSize', [ChunkSize_data], 'Deflate', 4);
    h5write(out_path,'/data',result);
    
    h5create(out_path,'/label',[Dataspace_label],'Datatype','uint8','ChunkSize', [ChunkSize_label],'Deflate', 1);
    h5write(out_path,'/label',label);
    
    h5create(out_path,'/pid',[Dataspace_pid],'Datatype','uint8','ChunkSize', [ChunkSize_pid],'Deflate', 1);
    h5write(out_path,'/pid',pid);
    
%     h5disp(out_path);
end

