function [HammingLoss,RankingLoss,OneError,Coverage,AveragePrecision,Macro_F1,Micro_F1] = BLS_PML(fileNames,fileFolder)
%TLDFS_PML Summary of this function goes here
%   Detailed explanation goes here
Dataset = strcat(fileFolder,fileNames);
load(Dataset);

% preprocess
n_fold = 10;
n_sample = size(data, 1);
n_test = round(n_sample / n_fold);
% data = zscore(data);%标准化函数

%LE
param.lambda1 = -3;
param.lambda2 = 1;
param.maxlters = 10;
param.stopPrecision = 1e-4;
param.mu = 1;
[~,numerical] = LE(partial_labels',data,param);


ratio = 0.7;
[~, lower_train_data] = FS(data', partial_labels',numerical,ratio);
data = lower_train_data';
partial_labels = numerical';


result = zeros(n_fold+2, 7); % save evaluation results
% n_fold validation and evaluation
for i = 1:n_fold
    fprintf('Data processing, Cross validation: %d\n', i);
    % split data
    start_idx = (i-1)*n_test + 1;
    if i == n_fold
        test_idx = start_idx : n_sample;
    else
        test_idx = start_idx:start_idx + n_test - 1;
    end
    II = 1:n_sample;
    train_idx = setdiff(II, test_idx);
    train_data = data(train_idx,:);
    train_p_target = partial_labels(:,train_idx);

    test_data = data(test_idx,:);
    test_target = target(:,test_idx);
    test_data(all(test_target==0,1),:) = [];
    test_target(:,all(test_target==0,1)) = [];

    % preprossing
    [Train_x,Test_x] = pre_zca(train_data,test_data);
    train_x = Train_x;test_x = Test_x;
    %one shot
    C = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
    N1=10;%feature nodes  per window
    N2=10;% number of windows of feature nodes
    N3=300;% number of enhancement nodes
    [HammingLoss,RankingLoss,OneError,Coverage,AveragePrecision,Macro_F1,Micro_F1,~,~] = bls_train(train_x,train_p_target',test_x,test_target,s,C,N1,N2,N3);       
      
   
    result(i,:) = [HammingLoss,RankingLoss,OneError,Coverage,AveragePrecision,Macro_F1,Micro_F1];

end
result(i+1,:) = mean(result(1:n_fold,:));
result(i+2,:) = std(result(1:n_fold,:));

RankingLoss = result(:,2);
OneError = result(:,3);
AveragePrecision = result(:,5);
Coverage = result(:,4);
HammingLoss = result(:,1);
Macro_F1 = result(:,6);
Micro_F1 = result(:,7);
end