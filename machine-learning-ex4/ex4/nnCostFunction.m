function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%---BSS Below is feed forward code---
  a1 = [ones(size(X,1),1) X];
  
  z2 = a1*Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(size(a2,1),1) a2];
  
  z3 = a2*Theta2';
  h = sigmoid(z3);
%---------------------------------  

%------BSS make y vector of 5000*1 to 5000*10.. for example each row is all zeros except for one 1 
%(depending on k value)
  yk = zeros(size(X,1), num_labels);
  for i=1:size(X,1)
    yk (i, y(i)) = 1;
  end
  
%--- BSS Without for loop you get 10*1 J vector. so compute J for each k (1 to 10) and then add all costs  
  for k = 1:num_labels
    
    Jk = (((-1)/m) * ((yk(:,k)'*(log(h(:,k)))) + ((1-yk(:,k))'*(log(1-h(:,k))))));
    J = J + Jk;
    
  end

%---Regularization code-----------------------------
  Theta1_temp = Theta1; 
  Theta1_temp(:,1)=0;
  
  Theta2_temp = Theta2;
  Theta2_temp(:,1)=0;
  
  reg = (lambda/(2*m))* ...
         (sum(sum(Theta1_temp.*Theta1_temp)) + sum(sum(Theta2_temp.*Theta2_temp)));
         
  J = J + reg;

%---Back Propagation---------------------------

  delta3 = h-yk;  
  delta2 = (delta3*Theta2(:,2:end)).*sigmoidGradient(z2);
  
%---Calculate Gradients------------------------

  Theta1_grad = (1/m)*(delta2'*a1);
  Theta2_grad = (1/m)*(delta3'*a2);
  
  %---Add Regularization for gradients----
  Theta1_grad = Theta1_grad + (lambda/m)*Theta1_temp;
  Theta2_grad = Theta2_grad + (lambda/m)*Theta2_temp;  
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
