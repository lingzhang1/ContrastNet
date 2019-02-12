% read data
mainpath = './data/modelnet40_ply_hdf5_2048';

data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

%     label_path = strcat( mainpath, '/', category, '/points_label', '/*.seg');
%     label_files = dir(label_path);

mkdir data/modelnet40_ply_hdf5_2048_cut;

num = 0;
for n=1:length(data_files)
    data_path = strcat( mainpath, '/',data_files(n).name);

%     h5disp(data_path);
    data = h5read(data_path,'/data');
    faceId = h5read(data_path,'/faceId');
    label = h5read(data_path,'/label');
    normal = h5read(data_path,'/normal');  

    x = length(data(:, 1, 1));
    y = length(data(1, :, 1));
    z = length(data(1, 1, :));

    for i = 1:z
        xyzPoints = data(:,:,11);
        xyzPoints = xyzPoints(1:3,:);
        xyzPoints = xyzPoints'; 
        xyzLabel = label(i);
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
            
            middle_axis = 3 * 2 - max_axis - min_axis;
            
            % mean
            M = mean(xyzPoints)          
            
            A1 = xyzPoints(:,max_axis) > M(max_axis) * (1 + 2 * 0.1 * (rand - 0.5));
            cut1 = xyzPoints(A1 ~= 0,:);
            A1 = cut1(:,middle_axis) > M(middle_axis) * (1 + 2 * 0.1 * (rand - 0.5));
            cut1 = cut1(A1 ~= 0,:);
%             length(cut1)
            
            A2 = xyzPoints(:,max_axis) <= M(max_axis) * (1 + 2 * 0.1 * (rand - 0.5));
            cut2 = xyzPoints(A2 ~= 0,:);
            A2 = cut2(:,middle_axis) > M(middle_axis) * (1 + 2 * 0.1 * (rand - 0.5));
            cut2 = cut2(A2 ~= 0,:);
%             length(cut2)
            
            A3 = xyzPoints(:,middle_axis) <= M(middle_axis) * (1 + 2 * 0.1 * (rand - 0.5));
            cut3 = xyzPoints(A3 ~= 0,:);
            A3 = cut3(:,max_axis) > M(max_axis) * (1 + 2 * 0.1 * (rand - 0.5));
            cut3 = cut3(A3 ~= 0,:);
%             length(cut3)
                        
            A4 = xyzPoints(:,middle_axis) <= M(middle_axis) * (1 + 2 * 0.1 * (rand - 0.5));
            cut4 = xyzPoints(A4 ~= 0,:);
            A4 = cut4(:,max_axis) <=  M(max_axis) * (1 + 2 * 0.1 * (rand - 0.5));
            cut4 = cut4(A4 ~= 0,:);            
%             length(cut4)
            
%             figure;
%             pcshow(cut1);
%             title('cut1');
%             figure;
%             pcshow(cut2);
%             title('cut2');
%             figure;
%             pcshow(cut3);
%             title('cut3');
%             figure;
%             pcshow(cut4);
%             title('cut4');            

        out_path = strcat('./data/modelnet40_ply_hdf5_2048_cut/',num2str(num),'_', data_files(n).name);
        num = num + 1;
% 
%         info = h5info(data_path);
% 
%         Dataspace_data = info.Datasets(1).Dataspace.Size;
%         ChunkSize_data = info.Datasets(1).ChunkSize;
% 
%         Dataspace_label = info.Datasets(3).Dataspace.Size;
%         ChunkSize_label = info.Datasets(3).ChunkSize;

%         h5create(out_path,'/cut1',[length(cut1) length(cut1(1,:)) 1],'Datatype','single','Deflate', 4);
        h5create(out_path,'/cut1',[length(cut1) length(cut1(1,:))],'Datatype','single');
        h5write(out_path,'/cut1',cut1);
        
        h5create(out_path,'/cut2',[length(cut2) length(cut2(1,:))],'Datatype','single');
        h5write(out_path,'/cut2',cut2);
   
        h5create(out_path,'/cut3',[length(cut3) length(cut3(1,:))],'Datatype','single');
        h5write(out_path,'/cut3',cut3);
        
        h5create(out_path,'/cut4',[length(cut4) length(cut4(1,:))],'Datatype','single');
        h5write(out_path,'/cut4',cut4);        
        
%         h5create(out_path,'/label',[1],'Datatype','uint8', 'Deflate', 1);
        h5create(out_path,'/label',[1],'Datatype','uint8');
        h5write(out_path,'/label',xyzLabel);

    %     h5disp(out_path);
    end

    processing = data_files(n).name
end
