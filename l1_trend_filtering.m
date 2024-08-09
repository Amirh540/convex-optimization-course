rng(1); 
n = 1000;
y = cumsum(randn(n, 1)); % Random walk

% Form second difference matrix
e = ones(n, 1);
D = spdiags([e -2*e e], 0:2, n-2, n);

lambda = 80;

cvx_begin
    variable x(n)
    minimize( 0.5 * sum_square(y - x) + lambda * norm(D * x, 1) )
cvx_end

figure;
plot(1:n, y, 'k:', 'LineWidth', 1.0); hold on;
plot(1:n, x, 'b-', 'LineWidth', 2.0); hold off;
xlabel('Time');
ylabel('Value');
legend('Original Signal', 'Estimated Trend');
title('L1 Trend Filtering');
