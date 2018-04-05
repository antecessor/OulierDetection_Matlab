%% Create Data
x1=1:.01:10;
x2=cos(x1)+unifrnd(0,.5,size(x1));
X=[x1',x2'];
%% Add Outlier
X(200,2)=1;
X(600,2)=0;
X(500,2)=0;
plot(X(:,1),X(:,2),'.')
hold on
%%
indexOutlier=OutlierDetectionAlgorithm(X,.01,21,3);
plot(X(indexOutlier,1),X(indexOutlier,2),'*k')

