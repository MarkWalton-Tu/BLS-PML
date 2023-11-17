clear ;clc
warning off;
addpath(genpath(pwd));

fileFolder = fullfile('.\data\');
fileName = 'YeastBP_1';
[HammingLoss,RankingLoss,OneError,Coverage,AveragePrecision,Macro_F1,Micro_F1] = BLS_PML(fileName,fileFolder);



