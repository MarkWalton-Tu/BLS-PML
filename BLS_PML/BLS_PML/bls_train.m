function [HammingLoss,RankingLoss,OneError,Coverage,AveragePrecision,Macro_F1,Micro_F1,predict_LD,bin_Pre_LD] = bls_train(train_x,train_y,test_x,test_y,s,C,N1,N2,N3)
% Learning Process of the proposed broad learning system
%Input: 
%---train_x,test_x : the training data and learning data 
%---train_y,test_y : the label 
%---We: the randomly generated coefficients of feature nodes
%---wh:the randomly generated coefficients of enhancement nodes
%----s: the shrinkage parameter for enhancement nodes
%----C: the regularization parameter for sparse regualarization
%----N11: the number of feature nodes  per window
%----N2: the number of windows of feature nodes


train_x = zscore(train_x')';
H1 = [train_x .1 * ones(size(train_x,1),1)];y=zeros(size(train_x,1),N2*N1);
for i=1:N2
    we=2*rand(size(train_x,2)+1,N1)-1;
    We{i}=we;
    A1 = H1 * we;A1 = mapminmax(A1);
    clear we;
beta1  =  sparse_bls(A1,H1,1e-3,50)';
beta11{i}=beta1;

T1 = H1 * beta1;


[T1,ps1]  =  mapminmax(T1',0,1);T1 = T1';
ps(i)=ps1;

y(:,N1*(i-1)+1:N1*i)=T1;
end

clear H1;
clear T1;


H2 = [y .1 * ones(size(y,1),1)];
if N1*N2>=N3
     wh=orth(2*rand(N2*N1+1,N3)-1);
else
    wh=orth(2*rand(N2*N1+1,N3)'-1)'; 
end
T2 = H2 *wh;
l2 = max(max(T2));
l2 = s/l2;


T2 = tansig(T2 * l2);
T3=[y T2];
clear H2;clear T2;
beta = (T3'  *  T3+eye(size(T3',1)) * (C)) \ ( T3'  *  train_y);

clear T3;

test_x = zscore(test_x')';
HH1 = [test_x .1 * ones(size(test_x,1),1)];

yy1=zeros(size(test_x,1),N2*N1);
for i=1:N2
    beta1=beta11{i};ps1=ps(i);
    TT1 = HH1 * beta1;
    TT1  =  mapminmax('apply',TT1',ps1)';

clear beta1; clear ps1;

yy1(:,N1*(i-1)+1:N1*i)=TT1;
end
clear TT1;clear HH1;
HH2 = [yy1 .1 * ones(size(yy1,1),1)]; 
TT2 = tansig(HH2 * wh * l2);TT3=[yy1 TT2];
clear HH2;clear wh;clear TT2;

x = TT3 * beta;
clear TT3;



% evalution
distribution  = softmax(x');
predict_LD = distribution';
bin_Pre_LD = binaryzation(softmax(predict_LD')',0.065);

bin_Pre_LD(bin_Pre_LD==0)=-1;
test_y(test_y==0)=-1;

HammingLoss=Hamming_loss(bin_Pre_LD',test_y);
Macro_F1 = MacroF1(bin_Pre_LD',test_y);
Micro_F1 = MicroF1(bin_Pre_LD',test_y);
OneError=One_error(predict_LD',test_y);
RankingLoss=Ranking_loss(predict_LD',test_y);
Coverage=coverage(predict_LD',test_y);
AveragePrecision=Average_precision(predict_LD',test_y);

