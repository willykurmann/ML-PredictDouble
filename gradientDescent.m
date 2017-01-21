%% Compute gradient descent until convergence
function [theta,  J_history] = gradientDescent(X, y, theta, alpha,  num_iters)

m = length(X);
J_history = zeros(num_iters,2);

for i=1:num_iters

    temp_theta0 = theta(1) - alpha * sum((X*theta)-y) / m;
    temp_theta1 = theta(2) - alpha * sum(((X*theta)-y) .* X(:,2)) / m;

    theta = [temp_theta0;temp_theta1];

    J = costFunction(X,y, theta);
    J_history(i,:) = [i J];

end

endfunction