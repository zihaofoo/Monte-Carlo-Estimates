
clear all

%% Variable definition for ODE solver
num_eig = 100;

[x_quad, weight_quad] = qrule(num_eig); 
x_quad = x_quad ./ 2 + 0.5; 
weight_quad = weight_quad ./ 2;

num_x = length(x_quad);
xgrid = x_quad;
mu_F = -1.0;
sigma_F = sqrt(0.2);
source = 5.0;           % s(x) = 5
rightbc = 1.0;          % u_r = 1

num_MC = 50;
mu = 1.0;
F = normrnd(mu_F, sigma_F);     % Sampling from Gaussian for F(w)


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

u_val_untruncated = zeros(num_MC, 1);

usol_untruncated = zeros(num_x, num_MC);


for i2 = 1:num_MC
    usol_untruncated(:, i2) = diffusioneqn(xgrid, F, k_untruncated (:, i2), source, rightbc);
end


%% Covariance Field 
u_mean_untruncated = mean(usol_untruncated, 2);
u_cov_untruncated_vec = usol_untruncated - u_mean_untruncated;

cov_untruncated_mat = u_cov_untruncated_vec * u_cov_untruncated_vec';

[X1, X2] = meshgrid(xgrid, xgrid);

%{
figure(1)
surf(X1, X2, cov_untruncated_mat)
xlim([0,1])
ylim([0,1])
xlabel('X Coordinate 1, X_1')
ylabel('X Coordinate 2, X_2')
zlabel('Covariance Field, C(X_1, X_2)')
title('Covariance field of untruncated Y')
axis('square')
%}

%%  Nystrom Method

W_root_mat = diag(sqrt(weight_quad)); 
[phi_mat, lambda_mat] = eig(W_root_mat * cov_untruncated_mat * W_root_mat);
eigen_vals_vec = real(diag(lambda_mat));              % Eigenvalues for KL expansion
eigen_vals_vec = eigen_vals_vec / eigen_vals_vec(1);

x_pos = 0.5;
[Y_r, eigen_vals_vec_Y] = Y_r_KL_expansion_old(x_pos, num_eig); 
eigen_vals_vec_Y = real(eigen_vals_vec_Y);
eigen_vals_vec_Y  = eigen_vals_vec_Y  / eigen_vals_vec_Y (1);


figure(2)
hold on
box on
n_vec = linspace(1, length(eigen_vals_vec), length(eigen_vals_vec)); 
plot(n_vec, eigen_vals_vec_Y, '^ - r')
plot(n_vec, eigen_vals_vec, 'v - k')

ylabel('Error in Truncation, \epsilon (-)');
xlabel('Stochastic dimension, n (-) ');
legend('Eigenvalues of Y(x,\omega)', 'Eigenvalues of u(x,\omega)')
axis('square')

    