% Sparse solution of a linear system of equations using l1-norm heuristic

% The threshold value below which we consider an element to be zero
delta = 1e-7;

% (100 equations and 200 unknowns)
m = 100;
n = 200;

A = randn(m, n);
x0 = sprandn(n, 1, 0.1); % Generate a sparse solution with sparsity 0.1
b = A * x0;

fprintf('Finding a sparse solution using l1-norm heuristic ...\n');
cvx_begin
    variable x_l1(n)
    minimize(norm(x_l1, 1))
    subject to
        A * x_l1 == b;
cvx_end

nnz_l1 = length(find(abs(x_l1) > delta));
fprintf('Found a solution in R^%d that has %d nonzeros using the l1-norm heuristic.\n', n, nnz_l1);

NUM_RUNS = 15;
nnzs = [];
W = ones(n, 1); 

disp([char(10) 'Log-based heuristic:']);
for k = 1:NUM_RUNS
    cvx_begin quiet
        variable x_log(n)
        minimize(sum(W .* abs(x_log)))
        subject to
            A * x_log == b;
    cvx_end

    nnz_log = length(find(abs(x_log) > delta));
    nnzs = [nnzs nnz_log];
    fprintf('   Found a solution with %d nonzeros...\n', nnz_log);

    W = 1 ./ (delta + abs(x_log));
end

nnz_log = length(find(abs(x_log) > delta));
fprintf('Found a solution in R^%d that has %d nonzeros using the log heuristic.\n', n, nnz_log);

figure;
plot(1:NUM_RUNS, nnzs, 'o-', 1:NUM_RUNS, repmat(nnzs(1), 1, NUM_RUNS), '--');
axis([1 NUM_RUNS 0 n])
xlabel('Iteration');
ylabel('Number of nonzeros (cardinality)');
legend('Log heuristic', 'l1-norm heuristic', 'Location', 'SouthEast');
title('Sparse Solution Finding Using l1-norm and Log-based Heuristics');
grid on;
