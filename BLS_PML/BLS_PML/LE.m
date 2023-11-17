function [noisy,clean] = LE(logicalLabel,features,param)
%LE Summary of this function goes here
%   Detailed explanation goes here

% e-NN similarity matrix A
D = mean2(pdist2(features,features));
e = D;
Idx = rangesearch(features,features,e);% D = pdist2(features,features);
GraphConnect = zeros(size(features,1),size(features,1));
for i = 1:size(features,1)
    GraphConnect(i,cell2mat(Idx(i,:))) = 1;%a = cell2mat(Idx(1,:));
end
GraphConnect = GraphConnect + GraphConnect';
GraphConnect(GraphConnect > 0) = 1;
sigma = 10;
A =  exp(-(L2_distance(features', features').^2) / (2 * sigma ^ 2));
A = A .* GraphConnect;
A = A - diag(diag(A));
A_hat = diag(sum(A,2));
G = A_hat - A;

[r,~] = corr(logicalLabel,logicalLabel,'type','Pearson');
r_hat = diag(sum(r,2));
C = r_hat-r;

[clean,noisy]=Opt(logicalLabel,param.lambda1,param.lambda2,G,C,param.maxlters,param.stopPrecision,param.mu);
%   Enhance non-zero elements
clean = clean.*logicalLabel;
clean = normal1(clean);





end

