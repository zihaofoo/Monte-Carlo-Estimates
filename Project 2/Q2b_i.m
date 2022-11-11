
clear all
clc

num_MC = 200;
n_dim = 4; 
p_deg = 2;

%% x = 0
x_loc = 0;

u_vec = solver_least_sq(n_dim, p_deg, num_MC, x_loc);

figure(1)
hold on
box on
histogram(u_vec, 30)
title('x = 0')
axis('square')

mu_PCE = mean(u_vec);
var_PCE = var(u_vec);

%% x = 0.5
x_loc = 0.5;

u_vec = solver_least_sq(n_dim, p_deg, num_MC, x_loc);

figure(2)
hold on
box on
histogram(u_vec, 30)
title('x = 0.5')
axis('square')
mu_PCE = mean(u_vec);
var_PCE = var(u_vec);

%% x = 0.75
x_loc = 0.75;

u_vec = solver_least_sq(n_dim, p_deg, num_MC, x_loc);

figure(3)
hold on
box on
histogram(u_vec, 30)
title('x = 0.75')
axis('square')

mu_PCE = mean(u_vec);
var_PCE = var(u_vec);

%% Mean Field
num_MC = 200;
n_dim = 4; 
p_deg = 3;
num_x = 50;
x_vec = linspace(0, 1, num_x); 
u_mean_vec = zeros(num_x, 1);
usol_LS = zeros(num_x, num_MC);

for i1 = 1:length(x_vec)
    u_vec = solver_least_sq(n_dim, p_deg, num_MC, x_vec(i1), num_x);
    u_mean_vec(i1) = mean(u_vec); 
    usol_LS(i1, :) = u_vec;
    i1
end

figure(4)
hold on
box on
plot(x_vec , u_mean_vec , 'r', 'LineWidth', 1.5)
xlabel('X coordinate, x')
ylabel('Mean field solution, u(x)')
axis('square')
legend('Least Square')
title('Mean field of least square approximation')

%% Covariance Field
u_cov_LS_vec = usol_LS - u_mean_vec;

cov_LS_mat = u_cov_LS_vec * u_cov_LS_vec';

[X1, X2] = meshgrid(x_vec, x_vec);

figure(5)
surf(X1, X2, cov_LS_mat)
xlim([0,1])
ylim([0,1])
xlabel('X Coordinate 1, X_1')
ylabel('X Coordinate 2, X_2')
zlabel('Covariance Field, C(X_1, X_2)')
title('Covariance field of Least Squares')
axis('square')


