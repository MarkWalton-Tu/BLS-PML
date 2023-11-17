function [Theta, lower_train_data] = FS(train_data, train_p_target, Y, ratio)


%   Inputs:
%   train_data: an D*M array; D is the number of features and M is the number of instances
%   tain_p_target: an M*Q arry; Q is the number of labels
%   ratio: the threshold of retained features
%   k: the number of neighbours
%
%   Outputs:
%   Theta: an M*d array; d is the reduced dimension
%   lower_train_data: an d*M array


M = size(train_data, 2);
Q = size(train_p_target, 2);

LabelSet = zeros(Q, M + 6);

for i = 1:M
    for j = 1:Q
        if train_p_target(i, j) == 1
            LabelSet(j, 1) = LabelSet(j, 1) + 1;
            t = LabelSet(j, 1);
            LabelSet(j, t + 1) = i;

        end
    end
end



K = zeros(M, M) ;
sigma = 0.5;
for i = 1:M
    for j = 1:M
        K(i, j) = exp(-sum((train_data(:, i) - train_data(:, j)).^2) / (2 * sigma * sigma));
    end
end



[Theta, lower_train_data] = Dimen_Red(K, Y, ratio, LabelSet);





end
