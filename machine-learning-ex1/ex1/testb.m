A = [1 2; 6 3; 5 10]
%fprintf('%n \n', A);

B = ones (1, size(A,2))

mu = zeros (1, size(A,2));
sigma = zeros (1, size(A,2));

for i = 1:size(A,2)
    
    mu(:,i) = mean (A(:,i));
    sigma(:,i) = std (A(:,i));
    A(:,i) = A(:,i) - mu(:,i);
    
end
muu = mu
sigg = sigma
C = A
D = C(:,1)./sigma(:,1)
E = C(:,2)./sigma(:,2)

for j = 1:size(A,2)
    C(:,j) = C(:,j)./sigma(:,j);
end
F = C