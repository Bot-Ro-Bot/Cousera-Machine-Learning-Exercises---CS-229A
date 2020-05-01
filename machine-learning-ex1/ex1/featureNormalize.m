function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2)); %2 signifies columns, mu =(1*2) matrix
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
temp_mu= [ mean(X(:,1)), mean(X(:,2))];
temp_sigma = [ std(X(:,1)), std(X(:,2))];
X_norm = [( X_norm(:,1).-temp_mu(1) )/temp_sigma(1) , ( X_norm(:,2).-temp_mu(2) )/temp_sigma(2)];
mu= temp_mu;
sigma= temp_sigma;
% ============================================================

end
