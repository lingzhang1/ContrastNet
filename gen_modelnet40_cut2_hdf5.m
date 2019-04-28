% read data
mainpath = './data/modelnet40_ply_hdf5_2048';

data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

mkdir data/modelnet40_ply_hdf5_2048_cut;

num = 0;
point_num = 512;
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

        out_path = strcat('./data/modelnet40_ply_hdf5_2048_cut/',num2str(num),'_', data_files(n).name);
        num = num + 1;

%%%%%%%%%%%%%%%%  random cut part of the object  %%%%%%%%%%%%%%%
        % create 15 ramdom plains
        count = 0;
        while count < 30
            % a * x + b * y + c * z = 0
            points = xyzPoints;
            a = (rand - 0.5) * 2;
            b = (rand - 0.5) * 2;
            c = (rand - 0.5) * 2;
            
            points(:, 1) = points(:, 1) * a;
            points(:, 2) = points(:, 2) * b;
            points(:, 3) = points(:, 3) * c;
            S = sum(points,2);
            
            A1 = S >= 0;
            cut1 = xyzPoints(A1 ~= 0,:);
            
            A2 = S < 0;
            cut2 = xyzPoints(A2 ~= 0,:);
            
%             figure;
%             pcshow(cut1);
%             title('cut1');
%             figure;
%             pcshow(cut2);
%             title('cut2');
            
            if (length(cut1(:, 1)) > point_num) && (length(cut2(:, 1)) > point_num)
                
                cut1 = cut1';
                cut1_path = strcat( '/cut',num2str(count*2 + 1));
                h5create(out_path, cut1_path,[length(cut1(:, 1)) length(cut1(1,:))],'Datatype','single');
                h5write(out_path,cut1_path ,cut1);
                
                cut2 = cut2';
                cut2_path = strcat( '/cut',num2str(count*2 + 2));
                h5create(out_path,cut2_path,[length(cut2(:, 1)) length(cut2(1,:))],'Datatype','single');
                h5write(out_path, cut2_path,cut2);
                
                count = count + 1;
                
            end
        end
        h5create(out_path,'/label',[1],'Datatype','uint8');
        h5write(out_path,'/label',xyzLabel);
%         h5disp(out_path);
    end
    num
    processing = data_files(n).name
end
