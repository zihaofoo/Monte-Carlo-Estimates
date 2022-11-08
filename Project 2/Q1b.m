
clear all

%% Variable definition for ODE solver
start_x = 0.0;
end_x = 1.0;
num_x = 21;           

mu_F = -1.0;
sigma_F = sqrt(0.2);
source = 5.0;           % s(x) = 5
rightbc = 1.0;          % u_r = 1

%% Monte Carlo simulation
num_MC = 50;

n_dim = 5;
p_deg = 3;
mu = 1.0;
num_sample = 1;

xgrid = linspace(start_x, end_x, num_x)';
F = normrnd(mu_F, sigma_F);     % Sampling from Gaussian for F(w)
k_vec = zeros(num_x, num_MC);
Y_vec = zeros(num_x, num_MC);

for i1 = 1:num_x
    k_vec(i1, :) = k_TD_PCE(xgrid(i1), n_dim, p_deg, mu, num_MC);
    
    for i2 = 1:num_MC
        [Y_vec(i1, i2), ~] = Y_r_KL_expansion(xgrid(i1), n_dim);
    end
end

k_Y_vec = exp(Y_vec);

%% Untruncated Y

cov_x_mat = zeros(num_x, num_x);
for i1 = 1:num_x
    for i2 = 1:num_x
        cov_x_mat(i1, i2) = Cov_func(xgrid(i1), xgrid(i2));
    end
end

R = mvnrnd(mu * ones(num_x, 1), cov_x_mat, num_MC)';
k_untruncated = exp(R);

%% ODE Solver
x_loc = 0.5;

u_val_PCE = zeros(num_MC, 1);
u_val_KL = zeros(num_MC, 1);
u_val_untruncated = zeros(num_MC, 1);

usol_PCE = zeros(num_x, num_MC);
usol_KL = zeros(num_x, num_MC);
usol_untruncated = zeros(num_x, num_MC);


for i2 = 1:num_MC
    usol_PCE(:, i2) = diffusioneqn(xgrid, F, k_vec(:, i2), source, rightbc);
    usol_KL(:, i2) = diffusioneqn(xgrid, F, k_Y_vec(:, i2), source, rightbc);
    usol_untruncated(:, i2) = diffusioneqn(xgrid, F, k_untruncated (:, i2), source, rightbc);

    x_coord = abs(xgrid - x_loc) < (0.5 * ((end_x - start_x) / num_x));      % Index of x = x_loc
    u_val_PCE(i2) = usol_PCE(x_coord, i2);
    u_val_KL(i2) = usol_KL(x_coord, i2);
    u_val_untruncated(i2) = usol_KL(x_coord, i2);

    i2
end

figure(1)
hold on
box on
histogram(u_val_PCE, 30)
title('PCE')
mu_PCE = mean(u_val_PCE)
var_PCE = var(u_val_PCE)
axis('square')

figure(2)
hold on
box on
histogram(u_val_KL, 30)
title('KL of Y')
mu_KL = mean(u_val_KL)
var_KL = var(u_val_KL)
axis('square')


figure(3)
hold on
box on
histogram(u_val_untruncated, 30)
title('Untruncated Y')
mu_untrunc= mean(u_val_untruncated)
var_untrunc = var(u_val_untruncated)
axis('square')


figure(4)
hold on
box on
for i1 = 1:10
    plot(xgrid, usol_PCE(:, i1), 'LineWidth', 1.5)
end
ylabel('u from PCE')
xlabel('x')

figure(5)
hold on
box on
for i1 = 1:10
    plot(xgrid, usol_KL(:, i1), 'LineWidth', 1.5)
end
ylabel('u from KL')
xlabel('x')


%% Mean Field
figure(6)
hold on
box on
u_mean_KL = mean(usol_KL, 2);
u_mean_PCE = mean(usol_PCE, 2);
u_mean_untruncated = mean(usol_untruncated, 2);

plot(xgrid, u_mean_untruncated, 'r', 'LineWidth', 1.5)
plot(xgrid, u_mean_KL, 'b', 'LineWidth', 1.5)
plot(xgrid, u_mean_PCE, 'k', 'LineWidth', 1.5)

xlabel('X coordinate, x')
ylabel('Mean field solution, u(x)')
axis('square')
legend('Untruncated', 'KL Expansion', 'PCE')
title('n = 5, p = 3')

%% Covariance Field 


u_cov_KL_vec = usol_KL - u_mean_KL;
u_cov_PCE_vec = usol_PCE - u_mean_PCE;
u_cov_untruncated_vec = usol_untruncated - u_mean_untruncated;

cov_KL_mat = u_cov_KL_vec * u_cov_KL_vec';
cov_PCE_mat = u_cov_PCE_vec * u_cov_PCE_vec';
cov_untruncated_mat = u_cov_untruncated_vec * u_cov_untruncated_vec';

[X1, X2] = meshgrid(xgrid, xgrid);

figure(1)
surf(X1, X2, cov_KL_mat)
xlim([0,1])
ylim([0,1])
xlabel('X Coordinate 1, X_1')
ylabel('X Coordinate 2, X_2')
zlabel('Covariance Field, C(X_1, X_2)')
title('Covariance field of KL')
axis('square')

figure(2)
surf(X1, X2, cov_PCE_mat)
xlim([0,1])
ylim([0,1])
xlabel('X Coordinate 1, X_1')
ylabel('X Coordinate 2, X_2')
zlabel('Covariance Field, C(X_1, X_2)')
title('Covariance field of PCE')
axis('square')

figure(3)
surf(X1, X2, cov_untruncated_mat)
xlim([0,1])
ylim([0,1])
xlabel('X Coordinate 1, X_1')
ylabel('X Coordinate 2, X_2')
zlabel('Covariance Field, C(X_1, X_2)')
title('Covariance field of untruncated Y')
axis('square')