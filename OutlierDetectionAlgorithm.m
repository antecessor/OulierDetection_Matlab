function OutlierIndex=OutlierDetectionAlgorithm(X,h,k,numberOutlier)
% X: each row is a data and col is features
% d: dimential of problem
% h: precision of distrubution
% numberOutlier: number of outlier you estimate
d=size(X,2);
%% create KNN Graph with K-D tree with less computional cost
Mdl = KDTreeSearcher(X);
[Sknn,DGraph] = knnsearch(Mdl,X,'K',k); %find K-nearest 
% find reverse-nearest
Srnn={};
for i=1:size(Sknn,1)
    [Srnn{i},~]=find(Sknn(1:end,2:end)==i);
   
end
% find Share-neareast
Ssnn={};
for i=1:size(Sknn,1)
    Ssnn{i}=[];
   for j=2:size(Sknn,2)
       [Srnntemp,~]=find(Sknn(1:end,2:end)==Sknn(i,j));
       Srnntemp=unique(Srnntemp);
       Ssnn{i}=[Ssnn{i}; Srnntemp];
   end    
end;
% S_All
S_All={};
for i=1:size(X,1)
    S_All{i}=[Sknn(i,2:end) Srnn{i}' Ssnn{i}'];
     S_All{i}=unique(S_All{i});
     S_All{i} = setdiff(S_All{i},i);
end

%% find distance euclidean of all member and create Px 
DistanceMatrice=dist(X');
Kernelh=@(h,d,DistanceMatrice,i,j) (1/(2*pi)^(d/2))*exp(-DistanceMatrice(i,j).^2/(2*h^2));
P=zeros(size(X,1),1);
for i=1:size(X,1)
    j=S_All{i}; % member of S_ALL
    summation=0;
    for p=1:numel(j)
        summation=summation+(Kernelh(h,d,DistanceMatrice,i,j(p)));
    end
    P(i)=(1/(1+numel(S_All{i})))*(1/h^d)*summation;
end
%% calcualte RDOS
for i=1:size(X,1)
     j=S_All{i};
     RDOS(i)=sum(P(j))/(numel(j)*P(i));
end
%% sort in descending way
[~,Index] = sort(RDOS,'descend');
OutlierIndex=Index(1:numberOutlier);
end