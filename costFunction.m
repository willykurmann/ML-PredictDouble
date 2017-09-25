%% Compute the cost function (Squared error function)
function [J,grad] = costFunction(X, y, theta)

m = length(X);

% compute cost
sqrErr = ((X*theta)-y).^2;
J = sum(sqrErr) / (2*m);

% compute the gradient
grad = (X' * (X*theta - y))/m;

endfunction