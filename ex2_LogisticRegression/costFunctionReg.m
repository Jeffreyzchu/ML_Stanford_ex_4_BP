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

for i= 1:m
	J = J - y(i) * log(sigmoid(X(i,:) * theta)) ...
		  - (1-y(i))*log(1- sigmoid(X(i,:) * theta));
end 
RegulaLoss =0;
for j = 2: length(theta)
    RegulaLoss = RegulaLoss + lambda*(theta(j))^2;
end 

J = J / m + RegulaLoss / (2*m);

n = length(theta);        %num of features
%% ======================right
grad  = 1/m* X'*(sigmoid(X*theta) - y) + lambda/m*theta;
grad(1) = 1/m* X(:,1)'*(sigmoid(X*theta) - y);

% %% ======================wrong
% for j = 1:n
%     if (j==1)
%         grad(j) = 1/m* X(:,j)'*(sigmoid(X*theta) - y(j));
%     else
%         grad(j) = 1/m* X(:,j)'*(sigmoid(X*theta) - y(j)) + lambda/m*theta(j);
%     end 
% end 
% 
% % =============================================================

end
