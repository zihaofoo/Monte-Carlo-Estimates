
clear all

%% Variable definition for ODE solver
start_x = 0.0;
end_x = 1.0;
num_x = 21;           
xgrid = linspace(start_x, end_x, num_x)';

mu_F = -1.0;
sigma_F = sqrt(0.2);
source = 5.0;           % s(x) = 5
rightbc = 1.0;          % u_r = 1

mu = 1.0;
x_pos = 0.5;
num_MC = 100000;    % M

%% Generate untruncated K's
cov_x_mat = zeros(num_x, num_x);
for i1 = 1:num_x
    for i2 = 1:num_x
        cov_x_mat(i1, i2) = Cov_func(xgrid(i1), xgrid(i2));
    end
end

R = mvnrnd(mu * ones(num_x, 1), cov_x_mat, num_MC)';
R_B = mvnrnd(mu * ones(num_x, 1), cov_x_mat, num_MC)';

k_untruncated_A = exp(R);
k_untruncated_B = exp(R_B);


%% Generate F's
F_A = normrnd(mu_F, sigma_F, [num_MC, 1]);     % Sampling from Gaussian for F(w)
F_B = normrnd(mu_F, sigma_F, [num_MC, 1]);     % Sampling from Gaussian for F(w)

%% Matrix formation
% Solving for Y_a


%% ODE Solver
x_loc = 0.5;

y_A = zeros(num_MC, 1);
y_B = zeros(num_MC, 1);
y_C1 = zeros(num_MC, 1);
y_C2 = zeros(num_MC, 1);

usol_A = zeros(num_x, num_MC);
usol_B = zeros(num_x, num_MC);
usol_C1 = zeros(num_x, num_MC);
usol_C2 = zeros(num_x, num_MC);

for i2 = 1:num_MC
    usol_A(:, i2) = diffusioneqn(xgrid, F_A(i2), k_untruncated_A(:, i2), source, rightbc);
    usol_B(:, i2) = diffusioneqn(xgrid, F_B(i2), k_untruncated_B(:, i2), source, rightbc);

    usol_C1(:, i2) = diffusioneqn(xgrid, F_A(i2), k_untruncated_B(:, i2), source, rightbc);
    usol_C2(:, i2) = diffusioneqn(xgrid, F_B(i2), k_untruncated_A(:, i2), source, rightbc);

    x_coord = abs(xgrid - x_loc) < (0.5 * ((end_x - start_x) / num_x));      % Index of x = x_loc

    y_A(i2) = usol_A(x_coord, i2);
    y_B(i2) = usol_B(x_coord, i2);
    y_C1(i2) = usol_C1(x_coord, i2);
    y_C2(i2) = usol_C2(x_coord, i2);
end

f_0_sq = mean(y_A)^2;

%% Sensitivity Analysis
S_denom = ((y_A' * y_A) / num_MC) - f_0_sq;

S_1_numer = ((y_A' * y_C1) / num_MC) - f_0_sq;
S_2_numer = ((y_A' * y_C2) / num_MC) - f_0_sq;
S_1_total_numer = (((y_B' * y_C1) / num_MC) - f_0_sq) ;
S_2_total_numer = (((y_B' * y_C2) / num_MC) - f_0_sq) ;


S_1_main = S_1_numer / S_denom;
S_2_main = S_2_numer / S_denom;

S_1_total = 1 - (S_1_total_numer / S_denom);
S_2_total = 1 - (S_2_total_numer / S_denom) ;


[S_1_main, S_2_main; S_1_total, S_2_total]
