function [Theta, lower_train_data] = Dimen_Red(K, Y, ratio, LabelSet)
%   Function DIMEN_RED 

M = size(Y, 1);         %   number of instances
Q = size(Y, 2);

%   First step: calculate M_matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M_j = zeros(M, Q);      %   restore every vector m_j in each column

for j = 1:Q
    m_j = zeros(M, 1);
    
    len = LabelSet(j, 1);
    index = LabelSet(j, 2:(len + 1));
    
    sum_Y_ij = sum(Y(index, j));
    
    for it = 1:M
        m_j(it, 1) = Y(index, j)' * K(it, index)';
    end
    
    m_j = m_j / sum_Y_ij;
    M_j(:, j) = m_j;
    
end

m_star = sum(K, 2);
m_star = m_star ./ M; 

M_matrix = zeros(M, M);
for j = 1:Q
    M_matrix = M_matrix + (M_j(:, j) - m_star) * (M_j(:, j) - m_star)';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%   second step: calculate N_matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LabelSet = gpuArray(single(LabelSet));
N_i = K;
N_i = gpuArray(single(N_i));
N_j = M_j;
N_j = gpuArray(single(N_j));
y=Y;
y = gpuArray(y);

N_matrix = gpuArray(single(zeros(M,M))) ; 
for j = 1:Q
    len = LabelSet(j, 1);
    index = LabelSet(j, 2:(len + 1));
    
    for it = 1:len
        i = index(it);
        N_matrix = N_matrix + y(i, j) * (N_i(:, i) - N_j(:, j)) * (N_i(:, i) - N_j(:, j))'; 
    end
end
N_matrix = gather(N_matrix);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%   third step: calculate Theta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[V, D] = eig(single(M_matrix),single(N_matrix) ); % 返回广义特征值的对角矩阵 D 和满矩阵 V，其列是对应的右特征向量，
                                   % 使得 A*V = B*V*D。



eigVectors = V;
eigValues = diag(D);         %   eigValue is a column vector

n = size(eigValues, 1);
for i = 1:n
    if((isreal(eigValues(i))==1) & (eigValues(i) > 0.0) & (isinf(eigValues(i)) == 0))       %real positive eigenvalue
        continue;
    end
    eigValues(i)=0.0;
end

[eigValues, order] = sort(eigValues, 'descend');
eigVectors = eigVectors(:,order);

reduced_dimension = ceil(ratio * Q);
Theta = eigVectors(:, 1:reduced_dimension);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%   last step: get lower_train_data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lower_train_data = Theta' * K;

end
