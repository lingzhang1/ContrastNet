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
        xyzPoints = data(:,:,i);
        xyzPoints = xyzPoints(1:3,:);
        xyzPoints = xyzPoints'; 
        xyzLabel = label(i);
        
%         figure;
%         pcshow(xyzPoints);
%         title('Original');

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

%             x = rand * 2; % 0 - 3
%             if (x >= 0) && (x < 1)
%                 max_axis = max_axis;
%             else
%                 max_axis = middle_axis;
%             end
            % mean
            M = mean(xyzPoints);          
               
            A1 = xyzPoints(:,max_axis) > M(max_axis);
            cut1 = xyzPoints(A1 ~= 0,:);
            cut1 = cut1';
            A2 = xyzPoints(:,max_axis) <= M(max_axis);
            cut2 = xyzPoints(A2 ~= 0,:); 
            cut2 = cut2';
%             figure;
%             pcshow(cut1);
%             title('cut1');
%             figure;
%             pcshow(cut2);
%             title('cut2');
           
        out_path = strcat('./data/modelnet40_ply_hdf5_2048_cut/',num2str(num),'_', data_files(n).name);
        num = num + 1;
        h5create(out_path,'/cut1',[length(cut1(:, 1)) length(cut1(1,:))],'Datatype','single');
        h5write(out_path,'/cut1',cut1);
        
        h5create(out_path,'/cut2',[length(cut2(:, 1)) length(cut2(1,:))],'Datatype','single');
        h5write(out_path,'/cut2',cut2);      
        
        h5create(out_path,'/label',[1],'Datatype','uint8');
        h5write(out_path,'/label',xyzLabel);

    %     h5disp(out_path);
    end
    num
    processing = data_files(n).name
end
