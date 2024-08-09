n = 1000; % Number of data points
m = 15;  % Degree of polynomial
assert(n > m + 1, 'n must be greater than m + 1');
x = linspace(0, 50, n)'; 
y =  sin(0.5*x) - cos(0.5*x) + randn(n, 1); 

X = ones(n, m+1);
for i = 1:m
    X(:, i+1) = x.^i;
end

% Cross-validation 
num_folds = 5;
cv = cvpartition(n, 'KFold', num_folds);
lambda_values = logspace(-1, 1.2, 15); % Range of lambda values to test
mse_values = zeros(length(lambda_values), 1);

for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    mse_fold = zeros(num_folds, 1);
    
    for fold = 1:num_folds
        % Training and validation sets
        train_idx = training(cv, fold);
        val_idx = test(cv, fold);
        
        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_val = X(val_idx, :);
        y_val = y(val_idx);
        
        % Solve the Ridge regression problem using CVX
        cvx_begin quiet
            variable b(m+1)
            minimize(norm(y_train - X_train*b, 2) + lambda * norm(b, 2))
        cvx_end
        
        % Compute MSE on validation set
        y_val_pred = X_val * b;
        mse_fold(fold) = mean((y_val - y_val_pred).^2);
    end
    
    mse_values(i) = mean(mse_fold);
end

[~, best_idx] = min(mse_values);
best_lambda = lambda_values(best_idx);

cvx_begin
    variable b(m+1)
    minimize(norm(y - X*b, 2) + best_lambda * norm(b, 2))
cvx_end


b_est = b;
y_est = X * b_est;

figure;
plot(x, y, 'bo', 'MarkerFaceColor', 'b'); hold on; % Original data
plot(x, y_est, 'r-', 'LineWidth', 2); % Fitted polynomial
xlabel('x');
ylabel('y');
legend('Original data', 'Fitted polynomial');
title(['Polynomial Regression with Ridge Regularization using CVX (\lambda = ', num2str(best_lambda), ')']);
grid on;

figure;
semilogx(lambda_values, mse_values, 'b-', 'LineWidth', 2);
xlabel('\lambda');
ylabel('Mean Squared Error');
title('Cross-Validation for Ridge Regularization');
grid on;
