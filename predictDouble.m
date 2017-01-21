%% ML - Preidct the double of a value using linear regression

%% ==================== Initialization ====================

% Clear and Close Figures
clear ; close all; clc

% Generate a training set (vector) of 10 values
X = rand(100,1);
y = X*2;
m = length(X);

% Initialize the parameters(theta) to zero
theta = zeros(2,1);

% Add a 0 column to X
X = [zeros(m,1) X];

% Plot data
figure(1);
title ("Arbitrary title");
xlabel ("X");
ylabel ("y (double X)");
f1_legends = ["original"];
plot(X(:,2),y);

%% ==================== Cost function ====================

% Compute the initial Cost
J = costFunction(X,y,theta);
fprintf("Initial cost: %f\n",  J);
fprintf("\n");

%% ==================== Gradient descent ====================

alpha = [0.01 0.03 0.05 0.1 0.3 0.5];

num_iters = 100;

for i=1:length(alpha)
    
    theta = gradientDescent(X,y,theta,alpha(i),num_iters);
    fprintf("Theta found by gradient descent: %f, %f˙\n",  theta(1),theta(2));

    % Plot the linear fit
    hold on;
    plot(X(:,2),(X*theta));
    f1_legends = [f1_legends;mat2str(alpha(i))];
    legend(f1_legends);
    
end

% predictions
fprintf("\n");
fprintf("For value 12.3,  we predict a result of %f\n",  [0 12.3]*theta);
fprintf("For value 5,  we predict a result of %f\n",  [0 5]*theta);