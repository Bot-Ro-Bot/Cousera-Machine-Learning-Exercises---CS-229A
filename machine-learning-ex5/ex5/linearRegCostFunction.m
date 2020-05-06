function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypo = X*theta; %X= 12*2 , theta= 2*1, hypo = 12*1
J_cost = (1/(2*m))  * sum((hypo -y).^2);
J_reg = (lambda/(2*m))  * sum(theta(2:end,:).^2);

J = J_cost +J_reg;


grad_theta_0 = (1/m)* sum(X(:,1).*(hypo-y));
grad_theta_rest = (1/m)* sum( X(:,2:end).*(hypo-y))' + (lambda/m).*theta(2:end,:);

grad = [grad_theta_0 , grad_theta_rest'];


% =========================================================================

grad = grad(:);

end
