% read data
mainpath = './data/modelnet40_ply_hdf5_2048';

data_path = strcat( mainpath, '/*.h5');
data_files = dir(data_path);

mkdir data/modelnet40_ply_hdf5_2048_pers;

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

        out_path = strcat('./data/modelnet40_ply_hdf5_2048_pers/',num2str(num),'_', data_files(n).name);
        num = num + 1;

%%%%%%%%%%%%%%%%  random cut part of the object  %%%%%%%%%%%%%%%
        % create 15 ramdom plains
        count = 0;
        while count < 29
            % a * x + b * y + c * z = 0
            a = (rand - 0.5) * 2;
            b = (rand - 0.5) * 2;
            c = (rand - 0.5) * 2;
            
            visiblePtInds = HPR(xyzPoints,[a, b, c], 1);
            pers = xyzPoints(visiblePtInds,:);

%             figure;
%             pcshow(pers);
%             title('pers');
%             
            if length(pers(:, 1)) > point_num
                pers = pers';
                pers_path = strcat( '/cut',num2str(count + 1));
                h5create(out_path, pers_path,[length(pers(:, 1)) length(pers(1,:))],'Datatype','single');
                h5write(out_path,pers_path ,pers);
                
                count = count + 1;

            end
        end
        xyzPoints = xyzPoints';
        h5create(out_path, '/cut30',[length(xyzPoints(:, 1)) length(xyzPoints(1,:))],'Datatype','single');
        h5write(out_path, '/cut30' ,xyzPoints);
                
        h5create(out_path,'/label',[1],'Datatype','uint8');
        h5write(out_path,'/label',xyzLabel);
%         h5disp(out_path);
    end
    num
    processing = data_files(n).name
end

function visiblePtInds=HPR(p,C,param)
% HPR - Using HPR ("Hidden Point Removal) method, approximates a visible subset of points 
% as viewed from a given viewpoint.
% Usage:
% visiblePtInds=HPR(p,C,param)
%
% Input:
% p - NxD D dimensional point cloud.
% C - 1xD D dimensional viewpoint.
% param - parameter for the algorithm. Indirectly sets the radius.
%
% Output:
% visiblePtInds - indices of p that are visible from C.
%
% This code was written by Sagi Katz
% sagikatz@tx.technion.ac.il
% Technion, 2006.
% For more information, see "Direct Visibility of Point Sets", Katz S., Tal
% A. and Basri R., SIGGRAPH 2007, ACM Transactions on Graphics, Volume 26, Issue 3, August 2007.
%
% This method is patent pending.
dim=size(p,2);
numPts=size(p,1);
p=p-repmat(C,[numPts 1]);%Move C to the origin
normp=sqrt(dot(p,p,2));%Calculate ||p||
R=repmat(max(normp)*(10^param),[numPts 1]);%Sphere radius
P=p+2*repmat(R-normp,[1 dim]).*p./repmat(normp,[1 dim]);%Spherical flipping
visiblePtInds=unique(convhulln([double(P);zeros(1,dim)]));%convex hull
visiblePtInds(visiblePtInds==numPts+1)=[];
end

