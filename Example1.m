clc
clear
close
%% Create Data
Number=100;
mu = [1 .2];
sigma = [.006 -.001; .001 .006];
R = chol(sigma);
z1 = repmat(mu,Number,1) + randn(Number,2)*R;
mu = [2.2 .8];
sigma = [.02 .01; -.01 .02];
R = chol(sigma);
z2 = repmat(mu,Number,1) + randn(Number,2)*R;
%% add outlier
z2(end,:)=[1.2,1];
z1(end,:)=[1,1.1];
z1(end-1,:)=[2,0];
%% plot
plot(z1(:,1),z1(:,2),'o')
hold on
plot(z2(:,1),z2(:,2),'o')
%%
X=[z1;z2];
indexOutlier=OutlierDetectionAlgorithm(X,.01,21,3);
plot(X(indexOutlier,1),X(indexOutlier,2),'*k')
