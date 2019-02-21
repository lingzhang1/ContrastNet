% read data
mainpath = './data/modelnet40_ply_hdf5_2048';

data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

mkdir data/modelnet40_ply_hdf5_2048_cut4;

num = 0;
point_num = 256;
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

        out_path = strcat('./data/modelnet40_ply_hdf5_2048_cut4/',num2str(num),'_', data_files(n).name);
        num = num + 1;

%%%%%%%%%%%%%%%%  random cut part of the object  %%%%%%%%%%%%%%%
        % create 15 ramdom plains
        count = 0;
        while count < 8
            % a * x + b * y + c * z = 0
            % plain one
            a = rand;
            b = rand * 2;
            c = rand * 3 * (-1);
            
            points = xyzPoints;
            points(:, 1) = points(:, 1) * a;
            points(:, 2) = points(:, 2) * b;
            points(:, 3) = points(:, 3) * c;
            S = sum(points,2);
            A1 = S >= 0;
            cut1 = xyzPoints(A1 ~= 0,:);
            A2 = S < 0;
            cut2 = xyzPoints(A2 ~= 0,:);
            
            % plain two         
            a = rand;
            b = rand * 2;
            c = rand * 3 * (-1);
            
            points = cut1;
            points(:, 1) = points(:, 1) * a;
            points(:, 2) = points(:, 2) * b;
            points(:, 3) = points(:, 3) * c;
            S = sum(points,2);
            A1 = S >= 0;
            cut_1 = cut1(A1 ~= 0,:);
            A2 = S < 0;
            cut_2 = cut1(A2 ~= 0,:);
            
            points = cut2;
            points(:, 1) = points(:, 1) * a;
            points(:, 2) = points(:, 2) * b;
            points(:, 3) = points(:, 3) * c;
            S = sum(points,2);
            A1 = S >= 0;
            cut_3 = cut2(A1 ~= 0,:);
            A2 = S < 0;
            cut_4 = cut2(A2 ~= 0,:);
            
%             figure;
%             pcshow(cut_1);
%             title('cut1');
%             figure;
%             pcshow(cut_2);
%             title('cut2');
%             figure;
%             pcshow(cut_3);
%             title('cut3');
%             figure;
%             pcshow(cut_4);
%             title('cut4');
            
            if (length(cut_1(:, 1)) > point_num) && (length(cut_2(:, 1)) > point_num) && (length(cut_3(:, 1)) > point_num) && (length(cut_4(:, 1)) > point_num)
                
                cut_1 = cut_1';
                cut1_path = strcat( '/cut',num2str(count*4 + 1));
                h5create(out_path, cut1_path,[length(cut_1(:, 1)) length(cut_1(1,:))],'Datatype','single');
                h5write(out_path,cut1_path ,cut_1);
                
                cut_2 = cut_2';
                cut2_path = strcat( '/cut',num2str(count*4 + 2));
                h5create(out_path,cut2_path,[length(cut_2(:, 1)) length(cut_2(1,:))],'Datatype','single');
                h5write(out_path, cut2_path,cut_2);
               
                cut_3 = cut_3';
                cut3_path = strcat( '/cut',num2str(count*4 + 3));
                h5create(out_path, cut3_path,[length(cut_3(:, 1)) length(cut_3(1,:))],'Datatype','single');
                h5write(out_path,cut3_path ,cut_3);
                
                cut_4 = cut_4';
                cut4_path = strcat( '/cut',num2str(count*4 + 4));
                h5create(out_path,cut4_path,[length(cut_4(:, 1)) length(cut_4(1,:))],'Datatype','single');
                h5write(out_path, cut4_path,cut_4);
                               
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
