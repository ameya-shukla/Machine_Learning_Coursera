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




m = length(y);
h_theta = (X*theta);
% 12*1 matrix, where X is 12*2 matrix after adding a bias and theta is 2*1

% J is regularized as a cost function
J = ( (1/(2*m)) *  (h_theta - y)' * (h_theta - y) )   +   ( (lambda/(2*m)) * (theta(2:length(theta)))' * (theta(2:length(theta))) );
% J = 303.993192.....(values)



thetaZero = theta;
thetaZero(1) = 0;
% Considering the first value of theta as 0 because it is a bias value


grad = ((1/m) * (h_theta - y)' * X) + lambda / m * thetaZero';
% grad = [-15.303016; 598.250744]....(values)

% =========================================================================

grad = grad(:);

end
