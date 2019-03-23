% read data
mainpath = './data/ShapeNetCore.v2';

% Get a list of all files and folders in this folder.
files = dir(mainpath);
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
folders = files(dirFlags)
flag = zeros(length(folders));
for i = 1:length(folders)
    if length(folders(i).name) > 4
        flag(i) = 1;
    end
end
class_folders = folders(flag ~= 0);
                                                                                                                                                               
mkdir data/ShapeNetCore_hdf5_cut;

num = 0;
point_num = 512;
for n=1:length(class_folders)
    class_path = strcat( mainpath, '/', class_folders(5).name);
    label = n-1;
    obj_folders = dir(class_path);
    
    flag = zeros(length(obj_folders));
    for i = 1:length(obj_folders)
        if length(obj_folders(i).name) > 4
            flag(i) = 1;
        end
    end
    obj_folders = obj_folders(flag ~= 0);
    
    for k=1:length(obj_folders)
        data_path = strcat(class_path,'/', obj_folders(k).name, '/models/model_normalized.obj'); 
        obj = readObj(data_path);
        
%      v: [7826×3 double]
%      vt: [566×3 double]
%      vn: [5778×3 double]
%      f: [1×1 struct]

        xyzPoints = obj.v;
        figure;
        pcshow(xyzPoints);
        title('Original v');
        
        xyzPoints = obj.vn;
        figure;
        pcshow(xyzPoints);
        title('Original vn');

        out_path = strcat('./data/ShapeNetCore_hdf5_cut/',num2str(num),'_', class_folders(n).name, '.h5');
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
            
            figure;
            pcshow(cut1);
            title('cut1');
            figure;
            pcshow(cut2);
            title('cut2');
            
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
        h5write(out_path,'/label',label);
%         h5disp(out_path);

    end
    class_folders(n).name
end

function obj = readObj(fname)
%
% obj = readObj(fname)
%
% This function parses wavefront object data
% It reads the mesh vertices, texture coordinates, normal coordinates
% and face definitions(grouped by number of vertices) in a .obj file 
% 
%
% INPUT: fname - wavefront object file full path
%
% OUTPUT: obj.v - mesh vertices
%       : obj.vt - texture coordinates
%       : obj.vn - normal coordinates
%       : obj.f - face definition assuming faces are made of of 3 vertices
%
% Bernard Abayowa, Tec^Edge
% 11/8/07
% set up field types
v = []; vt = []; vn = []; f.v = []; f.vt = []; f.vn = [];
fid = fopen(fname);
% parse .obj file 
while 1    
    tline = fgetl(fid);
    if ~ischar(tline),   break,   end  % exit at end of file 
     ln = sscanf(tline,'%s',1); % line type 
     %disp(ln)
    switch ln
        case 'v'   % mesh vertexs
            v = [v; sscanf(tline(2:end),'%f')'];
        case 'vt'  % texture coordinate
            vt = [vt; sscanf(tline(3:end),'%f')'];
        case 'vn'  % normal coordinate
            vn = [vn; sscanf(tline(3:end),'%f')'];
        case 'f'   % face definition
            fv = []; fvt = []; fvn = [];
            str = textscan(tline(2:end),'%s'); str = str{1};

           nf = length(findstr(str{1},'/')); % number of fields with this face vertices
           [tok str] = strtok(str,'//');     % vertex only
            for k = 1:length(tok) fv = [fv str2num(tok{k})]; end

            if (nf > 0) 
            [tok str] = strtok(str,'//');   % add texture coordinates
                for k = 1:length(tok) fvt = [fvt str2num(tok{k})]; end
            end
            if (nf > 1) 
            [tok str] = strtok(str,'//');   % add normal coordinates
                for k = 1:length(tok) fvn = [fvn str2num(tok{k})]; end
            end
             f.v = [f.v; fv]; f.vt = [f.vt; fvt]; f.vn = [f.vn; fvn];
    end
end
fclose(fid);
% set up matlab object 
obj.v = v; obj.vt = vt; obj.vn = vn; obj.f = f;
end


