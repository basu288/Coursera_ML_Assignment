function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       


  h = X*all_theta'; % --- X is 5000x401 --- all_theta is 10x401 --- h is 5000x10
  
  % Get max of each row of h, so that max's index will be the predicted number (1-10)
  % for example let 1 row of h be -3, -5, -6, 8, -1, 0, -4, -5, -2, -4.. here 8 is highest probability number
  % Index of 8 is 4 -- so predicted number is 4
  
  % [max_num ind] = max (matrix) returns max of each column and returns idex of the row in that column. 
  % Since we want to get max of each row of h, let's flip h and get max of each colum and get its index, then
  % flip the indices vector

  [max_nums, indices] = max(h');

  p = indices';  


% =========================================================================


end
