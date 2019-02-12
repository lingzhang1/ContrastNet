% read data
mainpath = './data/modelnet40_ply_hdf5_2048';

data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

%     label_path = strcat( mainpath, '/', category, '/points_label', '/*.seg');
%     label_files = dir(label_path);

mkdir data/modelnet40_ply_hdf5_2048 cut;

for n=1:length(data_files)
    data_path = strcat( mainpath, '/',data_files(n).name);
%  data_files(n).name

%     h5disp(data_path);
    data = h5read(data_path,'/data');
    faceId = h5read(data_path,'/faceId');
    label = h5read(data_path,'/label');
    normal = h5read(data_path,'/normal');  

    x = length(data(:, 1, 1));
    y = length(data(1, :, 1));
    z = length(data(1, 1, :));
    result = zeros(x,y,z);
    nan_num = 0;
    for i = 1:z
        xyzPoints = data(:,:,400);
        xyzPoints = xyzPoints(1:3,:);
        xyzPoints = xyzPoints';
        
        figure;
        pcshow(xyzPoints);
        title('Original');

%%%%%%%%%%%%%%%%  reduction based on label  %%%%%%%%%%%%%%%

    %         label_path = strcat( mainpath, '/', category, '/points_label/', label_files(n).name);
    %         lables = load(label_path);
    %         A = (lables ~= 2);

%%%%%%%%%%%%%%%%  cut part of the object bansed on axis  %%%%%%%%%%%%%%%

            [Max_v,Max_i] = max(xyzPoints);
            [Min_v,Min_i] = min(xyzPoints);
            Range_value = Max_v - Min_v;
            [max_val, max_axis] = max(Range_value);
            [min_val, min_axis] = min(Range_value);
            % mean
            M = mean(xyzPoints);
            
%             A1 = xyzPoints(:,max_axis) > (Min_v(max_axis) + max_val * 0.5);
%             cut1 = xyzPoints(A1 ~= 0,:);
%             A1 = cut1(:,min_axis) > (Min_v(min_axis) + min_val * 0.5);
%             cut1 = cut1(A1 ~= 0,:);
            
            A1 = xyzPoints(:,max_axis) > M(max_axis);
            cut1 = xyzPoints(A1 ~= 0,:);
            A1 = cut1(:,min_axis) > M(min_axis);
            cut1 = cut1(A1 ~= 0,:);
            
            A2 = xyzPoints(:,max_axis) <= M(max_axis);
            cut2 = xyzPoints(A2 ~= 0,:);
            A2 = cut2(:,min_axis) > M(min_axis);
            cut2 = cut2(A2 ~= 0,:);
            
            A3 = xyzPoints(:,min_axis) <= M(min_axis);
            cut3 = xyzPoints(A3 ~= 0,:);
            A3 = cut3(:,max_axis) > M(max_axis);
            cut3 = cut3(A3 ~= 0,:);
                        
            A4 = xyzPoints(:,min_axis) <= M(min_axis);
            cut4 = xyzPoints(A4 ~= 0,:);
            A4 = cut4(:,max_axis) <=  M(max_axis);
            cut4 = cut4(A4 ~= 0,:);            
            
            % nomalize to 0 - 1
%             cut1 = (xyzPoints(A1 ~= 0,:) - Min_v(max_axis)) / max_val;
%             cut2 = (xyzPoints(A2 ~= 0,:) - Min_v(max_axis)) / max_val;
%             cut3 = (xyzPoints(A3 ~= 0,:)- Min_v(min_axis)) /  min_val;
%             cut4 = (xyzPoints(A4 ~= 0,:) - Min_v(min_axis)) /  min_val;
            
            figure;
            pcshow(cut1);
            title('cut1');
            figure;
            pcshow(cut2);
            title('cut2');
            figure;
            pcshow(cut3);
            title('cut3');
            figure;
            pcshow(cut4);
            title('cut4');            

%%%%%%%%%%%%%%%%  get normals  %%%%%%%%%%%%%%%
%         ptCloud = pointCloud(xyzPoints);
%         normals = pcnormals(ptCloud, 10);
%         
%         [row, col] = find(isnan(normals));
%         if length(row) ~= 0
%             xyzPoints(row,:) = xyzPoints(row-1,:);
%             normals(row,:) = normals(row-1,:);
%         end
%         [row, col] = find(isnan(normals));
%         nan_num = nan_num + length(row);

%%%%%%%%%%%%%%%%  show normals  %%%%%%%%%%%%%%%
%         figure;
%         pcshow(ptCloud);
%         title('Cut');
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
%         coords_normal = cat(2, xyzPoints, normals);
%         coords_normal = coords_normal';
        result(:,:,i) = cut1';
    end

%   
    processing = data_files(n).name
%     nan_num
%     Max = max(result,[],1,'includenan');
%     Max = max(Max,[],2,'includenan');
%     Max = max(Max,[],3,'includenan')
%     Min = min(result,[],1,'includenan');
%     Min = min(Min,[],2,'includenan');
%     Min = min(Min,[],3,'includenan')

    out_path = strcat( mainpath, '/coords_normal/',data_files(n).name);

    info = h5info(data_path);

%     Dataspace_data = info.Datasets(1).Dataspace.Size;
    ChunkSize_data = info.Datasets(1).ChunkSize;

    Dataspace_faceId = info.Datasets(2).Dataspace.Size;
    ChunkSize_faceId = info.Datasets(2).ChunkSize;

    Dataspace_label = info.Datasets(3).Dataspace.Size;
    ChunkSize_label = info.Datasets(3).ChunkSize;

    Dataspace_normal = info.Datasets(4).Dataspace.Size;
    ChunkSize_normal = info.Datasets(4).ChunkSize;

    h5create(out_path,'/data',[x y z],'Datatype','single','ChunkSize', [ChunkSize_data], 'Deflate', 4);
    h5write(out_path,'/data',result);

    h5create(out_path,'/faceId',[Dataspace_faceId],'Datatype','int32','ChunkSize', [ChunkSize_faceId],'Deflate', 1);
    h5write(out_path,'/faceId',faceId);

    h5create(out_path,'/label',[Dataspace_label],'Datatype','uint8','ChunkSize', [ChunkSize_label],'Deflate', 1);
    h5write(out_path,'/label',label);

    h5create(out_path,'/normal',[Dataspace_normal],'Datatype','single','ChunkSize', [ChunkSize_normal],'Deflate', 4);
    h5write(out_path,'/normal',normal);

%     h5disp(out_path);
end
