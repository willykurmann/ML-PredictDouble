%% ML - Preidct the double of a value using linear regression

%% ==================== Initialization ====================

% Clear and Close Figures
clear ; close all; clc

% Generate a training set (vector) of 10 values
X = rand(100,1);
y = X*2;
m = length(X);

% Initialize the parameters(theta) to zero
initial_theta = zeros(2,1);

% Add a 0 column to X
X = [ones(m,1) X];

% Plot data
figure(1);
title ("Arbitrary title");
xlabel ("X");
ylabel ("y (double X)");
f1_legends = ["original"];
plot(X(:,2),y);

%% ==================== Cost function ====================

% Compute the initial Cost
J = costFunction(X,y,initial_theta);
fprintf("Computing initial cost...\n");
fprintf("%f\n\n",  J);
f2_legends = "";

%% ==================== Gradient descent ====================

fprintf('Running Gradient Descent ...\n')
alpha = [0.05 0.1 0.3 0.5 0.6 0.8 1];
num_iters = 500;

for i=1:length(alpha)
    
    [theta, J_history] = gradientDescent(X, y, initial_theta, alpha(i), num_iters);
    fprintf("Theta(alpha=%.2f): %.5f, %.5f˙| cost: %.5f\n", alpha(i), theta(1),theta(2), costFunction(X,y,theta));

    % Plot the linear fit
    figure(1);
    hold on;
    plot(X(:,2),(X * theta));
    f1_legends = [f1_legends;mat2str(alpha(i))];
    legend(f1_legends);

    % Plot cost history
    figure(2);
    hold on;
    plot(J_history(:,1),J_history(:,2));
    f2_legends = [f2_legends;mat2str(alpha(i))];
    legend(f2_legends);
end

% predictions
fprintf("\n");
fprintf("For value 12.3,  we predict a result of %f\n",  [1 12.3] * theta);
fprintf("For value 5,  we predict a result of %f\n",  [1 5] * theta);

%% ==================== Optimizing using fminunc ====================
fprintf("\nOptimizing using fminunc...\n");
options = optimset('GradObj', 'on', 'MaxIter', 100);
[theta, cost,info, o] = fminunc(@(t)(costFunction(X, y, t)), initial_theta, options);
fprintf("Theta: %.5f, %.5f˙| cost: %.5f\n | iterations: %i", theta(1),theta(2), cost, o.iterations);

%% ==================== Computing theta using normal equation ====================
fprintf("\nComputing theta using normal equation...\n");
theta = pinv(X) * y;
fprintf("Theta: %.5f, %.5f˙| cost: %.5f\n", theta(1),theta(2), costFunction(X,y,theta));
