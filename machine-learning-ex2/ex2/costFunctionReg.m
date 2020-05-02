function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



%{

yo code chaldaina hai

hypo = sigmoid(X*theta);
hypo_grad_rest = sigmoid(X(:,2:end)*theta(2:end));

grad_0 = (1/m)* sum( X(:,1).* (hypo-y));
grad_rest = (1/m)* ( sum( X(:,2:end).* (hypo-y)) )' + (lambda/m).*theta(2:end);


cost_term = - (1 /m)* sum(y.*log(hypo)+ (1-y).*log(1- hypo) ) ;
reg_term = (lambda / (2 *m) ) * sum(theta.*theta) ;
J = cost_term + reg_term;


grad=([grad_0,grad_rest'])';


%}





hypo= sigmoid(X*theta); %hypothesis function

%hypo_grad_rest = sigmoid(X(:,2:end)*theta(2:end));

J_cost_term = - (1 /m)* sum( y.*log(hypo)+ (1-y).*log(1- hypo) ) ;
J_reg_term = (lambda / (2 *m) ) * sum(theta(2:end).^2) ;
J = J_cost_term + J_reg_term;

grad_cost_term = (1/m)*( (X(:,2:end)'*(hypo-y)));
grad_reg_term  = (lambda/m)*theta(2:end);

grad_theta_0 = (1/m)*( (X(:,1)'*(hypo-y))); %calculating gradient for oonly theta_0
grad_theta_left= grad_cost_term+grad_reg_term;

grad= [ grad_theta_0; grad_theta_left];


% =============================================================

end
