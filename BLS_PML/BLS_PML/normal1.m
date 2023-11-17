function out = normal1(A)
%NORMAL1 Summary of this function goes here
%   Detailed explanation goes here
s = sum(A,2);
[h,l] = size(A); 
for j = 1:h
    for i = 1:l
        A(j,i) = A(j,i)/s(j);
    end
end
out = A;
end

