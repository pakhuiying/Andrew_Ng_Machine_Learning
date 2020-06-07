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

z = theta'*X'; %z = OT*x
h = sigmoid(z); %1xm
log_h = log(h);%1xm
log_h_prime = log(ones(size(h))-h);%1xm
J = (-1/m)*(log_h*y + log_h_prime*(ones(m,1)-y)) + lambda/(2*m)*sum(theta(2:end).^2); %omit the first theta
grad(1,:) = ((h-y')*X(:,1))'*(1/m); %compute first column which is the ones w/o regularisation
grad(2:end,:) = ((h-y')*X(:,2:end))'*(1/m) + lambda/m*theta(2:end,:); %same dimension as theta -->3*1 %omit the first column of X, which is the ones



% =============================================================

end
