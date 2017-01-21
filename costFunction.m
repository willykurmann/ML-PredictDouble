%% Compute the cost function (Squared error function)
function J = costFunction(X, y, theta)
   
m = length(X);

sqrErr = ((X*theta)-y).^2;
J = sum(sqrErr) / (2*m);

endfunction