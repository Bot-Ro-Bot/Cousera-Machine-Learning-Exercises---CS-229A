function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
iterations = length(C_values); 

error = zeros(iterations,iterations);

for i = 1: iterations
  for j = 1: iterations
     %selected values of C and sigma haru lai train gareko
     % Train the SVM
     model= svmTrain(X, y, C_values(i), @(x1, x2) gaussianKernel(x1, x2, sigma_values(j)));
     %trained model bata chai predict garne cross validation set ko data haru                             
     predictions = svmPredict(model, Xval);
     %error nikalne tarika cross validation ko predicted output bata by the model
     error(i,j) = mean(double(predictions ~= yval)) ;
     fprintf('Training Model : (%d,%d)\n',i,j);
  endfor
endfor


[h, row]= min(min(error,[],2));
[h, column]= min(min(error,[],1));

C = C_values(row);
sigma = sigma_values(column);

%C = 1;
%sigma = 0.1;
%Value of C : 1
%Value of sigma : 0.1
% =========================================================================

end
