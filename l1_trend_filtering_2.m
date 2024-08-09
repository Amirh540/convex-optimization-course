rng(1); 
n = 1000;
y = cumsum(randn(n, 1)); % Random walk

% Form second difference matrix
e = ones(n, 1);
D = spdiags([e -2*e e], 0:2, n-2, n);

lambda_values = [0.1, 10,50, 100, 500];

figure;
hold on;
colors = lines(length(lambda_values));
for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    cvx_begin quiet
        variable x(n)
        minimize( 0.5 * sum_square(y - x) + lambda * norm(D * x, 1) )
    cvx_end
    
    plot(1:n, x, 'LineWidth', 2.0, 'Color', colors(i,:), 'DisplayName', ['\lambda = ' num2str(lambda)]);
end
plot(1:n, y, 'k:', 'LineWidth', 1.0, 'DisplayName', 'Original Signal');
hold off;
xlabel('Time');
ylabel('Value');
legend('show');
title('L1 Trend Filtering with Different \lambda Values');
grid on;
