% read data
mainpath = './data/modelnet40_ply_hdf5_2048';

data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

mkdir data/modelnet40_ply_hdf5_2048_cut8;

num = 0;
point_num = 128;
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

        out_path = strcat('./data/modelnet40_ply_hdf5_2048_cut8/',num2str(num),'_', data_files(n).name);
        num = num + 1;

%%%%%%%%%%%%%%%%  random cut part of the object  %%%%%%%%%%%%%%%
        % create count ramdom plains
        
        iteration = 0;
        count = 0;
        while count < 1
            % a * x + b * y + c * z = 0
            % plain one
            a = (rand - 0.5) * 2;
            b = (rand - 0.5) * 2;
            c = (rand - 0.5) * 2;
            
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
            a = (rand - 0.5) * 2;
            b = (rand - 0.5) * 2;
            c = (rand - 0.5) * 2;
            
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
            
            % plain three
            a = (rand - 0.5) * 2;
            b = (rand - 0.5) * 2;
            c = (rand - 0.5) * 2;
            
            points = cut_1;
            points(:, 1) = points(:, 1) * a;
            points(:, 2) = points(:, 2) * b;
            points(:, 3) = points(:, 3) * c;
            S = sum(points,2);
            A1 = S >= 0;
            cut_11 = cut_1(A1 ~= 0,:);
            A2 = S < 0;
            cut_22 = cut_1(A2 ~= 0,:);
            
            points = cut_2;
            points(:, 1) = points(:, 1) * a;
            points(:, 2) = points(:, 2) * b;
            points(:, 3) = points(:, 3) * c;
            S = sum(points,2);
            A1 = S >= 0;
            cut_33 = cut_2(A1 ~= 0,:);
            A2 = S < 0;
            cut_44 = cut_2(A2 ~= 0,:);
            
            points = cut_3;
            points(:, 1) = points(:, 1) * a;
            points(:, 2) = points(:, 2) * b;
            points(:, 3) = points(:, 3) * c;
            S = sum(points,2);
            A1 = S >= 0;
            cut_55 = cut_3(A1 ~= 0,:);
            A2 = S < 0;
            cut_66 = cut_3(A2 ~= 0,:);  
       
            points = cut_4;
            points(:, 1) = points(:, 1) * a;
            points(:, 2) = points(:, 2) * b;
            points(:, 3) = points(:, 3) * c;
            S = sum(points,2);
            A1 = S >= 0;
            cut_77 = cut_4(A1 ~= 0,:);
            A2 = S < 0;
            cut_88 = cut_4(A2 ~= 0,:);
            
%             figure;
%             pcshow(cut_11);
%             title('cut1');
%             figure;
%             pcshow(cut_22);
%             title('cut2');
%             figure;
%             pcshow(cut_33);
%             title('cut3');
%             figure;
%             pcshow(cut_44);
%             title('cut4');
%             figure;
%             pcshow(cut_55);
%             title('cut5');
%             figure;
%             pcshow(cut_66);
%             title('cut6');
%             figure;
%             pcshow(cut_77);
%             title('cut7');
%             figure;
%             pcshow(cut_88);
%             title('cut8');
            iteration = iteration + 1;
            if (length(cut_11(:, 1)) > point_num) && (length(cut_22(:, 1)) > point_num) && (length(cut_33(:, 1)) > point_num) && (length(cut_44(:, 1)) > point_num) && (length(cut_55(:, 1)) > point_num) && (length(cut_66(:, 1)) > point_num) && (length(cut_77(:, 1)) > point_num) && (length(cut_88(:, 1)) > point_num)
                cut_11 = cut_11';
                cut1_path = strcat( '/cut',num2str(count*8 + 1));
                h5create(out_path, cut1_path,[length(cut_11(:, 1)) length(cut_11(1,:))],'Datatype','single');
                h5write(out_path,cut1_path ,cut_11);
                
                cut_22 = cut_22';
                cut2_path = strcat( '/cut',num2str(count*8 + 2));
                h5create(out_path,cut2_path,[length(cut_22(:, 1)) length(cut_22(1,:))],'Datatype','single');
                h5write(out_path, cut2_path,cut_22);
               
                cut_33 = cut_33';
                cut3_path = strcat( '/cut',num2str(count*8 + 3));
                h5create(out_path, cut3_path,[length(cut_33(:, 1)) length(cut_33(1,:))],'Datatype','single');
                h5write(out_path,cut3_path ,cut_33);
                
                cut_44 = cut_44';
                cut4_path = strcat( '/cut',num2str(count*8 + 4));
                h5create(out_path,cut4_path,[length(cut_44(:, 1)) length(cut_44(1,:))],'Datatype','single');
                h5write(out_path, cut4_path,cut_44);
                               
                cut_55 = cut_55';
                cut5_path = strcat( '/cut',num2str(count*8 + 5));
                h5create(out_path, cut5_path,[length(cut_55(:, 1)) length(cut_55(1,:))],'Datatype','single');
                h5write(out_path,cut5_path ,cut_55);
                
                cut_66 = cut_66';
                cut6_path = strcat( '/cut',num2str(count*8 + 6));
                h5create(out_path,cut6_path,[length(cut_66(:, 1)) length(cut_66(1,:))],'Datatype','single');
                h5write(out_path, cut6_path,cut_66);
               
                cut_77 = cut_77';
                cut7_path = strcat( '/cut',num2str(count*8 + 7));
                h5create(out_path, cut7_path,[length(cut_77(:, 1)) length(cut_77(1,:))],'Datatype','single');
                h5write(out_path,cut7_path ,cut_77);
                
                cut_88 = cut_88';
                cut8_path = strcat( '/cut',num2str(count*8 + 8));
                h5create(out_path,cut8_path,[length(cut_88(:, 1)) length(cut_88(1,:))],'Datatype','single');
                h5write(out_path, cut8_path,cut_88);
                
                h5create(out_path,'/label',[1],'Datatype','uint8');
                h5write(out_path,'/label',xyzLabel);
                
                count = count + 1;
            else
                if iteration > 500000
                    i
                    i = i + 1;
                    break;
                end
            end
        end
%         h5disp(out_path);
    end
    num
    processing = data_files(n).name
end
