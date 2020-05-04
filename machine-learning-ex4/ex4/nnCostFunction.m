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


%..............forward propagation ko code..............% 
X  = [ones(m,1) X]; %adding the +1 term , X= 5000*401
a2 = sigmoid(Theta1*X')'; ,% a2= 5000*25
a2 = [ones(size(a2,1),1) a2]; %a2= 5000*26
a3 = sigmoid(Theta2*a2'); %a3= 10*5000
hypo = a3'; %hypo = 5000*10
%........................................................%


%......code for converting y into 5000*10 matrix..........%
temp_y= zeros(m,num_labels);
for i=1:m
  temp_y(i,y(i))=1;
endfor
%........................................................%


%..........cost term calculation (without reg).................%
J_cost = sum(temp_y.*log(hypo)+ (1-temp_y).*log(1-hypo) );
J_cost = -(1/m)*sum(J_cost,2); %sum along the row ko laagi 2 rakheko
%........................................................%


%.................reg term calculation..........................%
J_reg = sum(sum( Theta1(:,2:end).^2)) + sum( sum( Theta2(:,2:end).^2)); %zero waala term hatayeko
J_reg = ( lambda / (2 *m) ) * J_reg;
%........................................................%

%final cost 
J= J_cost + J_reg;     


%..............back propagation ko code..............% 
for i =1:m
  %step 1
  a1 = X(i,:);   % X= 5000*401 ,  a1= 1*401 
  %a1 = [1 , a1];  %a1= 1*401
  a2 = sigmoid(a1*Theta1');  %a1= 1*401, %Theta1 = 25*401 -- %a2= 1*25
  a2 = [1 , a2];  %a2= 1*26
  a3 = sigmoid(a2*Theta2');; %a2= 1*26, %Theta2 = 10*26   -- %a3= 1*10
 
 %step 2
  d3 = a3 - temp_y(i,:);  %d3= 1*10 , error term for ith term in layer 3

  %step 3
  d2 = (d3*Theta2).*(a2.*(1-a2));  %d2= 1*26 , error term for ith term in layer 2
  d2 = d2(2:end);     %d2= 1*25

  %step 4
  Theta1_grad = Theta1_grad + d2'*a1;
  Theta2_grad = Theta2_grad + d3'*a2;
 
endfor

  %step 5
  
  %{ without regularization
  Theta1_grad = (Theta1_grad/m);
  Theta2_grad = (Theta2_grad/m);
  %}
  
  %add regularization term
  
  Theta1_grad_cost_term = (1/m)* (Theta1_grad(:,));
  Theta2_grad_cost_term = (1/m)* (Theta2_grad(:,1));
  
  Theta1_grad_reg_term = (Theta1_grad(:,2:end)/m) + (lambda/m)*Theta1_grad(:,2:end);
  Theta2_grad_reg_term = (Theta2_grad(:,2:end)/m) + (lambda/m)*Theta2_grad(:,2:end);
  
  Theta1_grad = [Theta1_grad_cost_term ,Theta1_grad_reg_term] ;
  Theta2_grad = [Theta2_grad_cost_term , Theta2_grad_reg_term];
  
  

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
