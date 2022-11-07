
clear all
clc

num_MC = 100;
n_dim = 4; 
p_deg = 2;

%% x = 0
x_loc = 0;

u_vec = solver_least_sq(n_dim, p_deg, num_MC, x_loc);

figure(1)
hold on
box on
histogram(u_vec, 30)
title('First')
mu_PCE = mean(u_vec);
var_PCE = var(u_vec);

%% x = 0.5
x_loc = 0.5;

u_vec = solver_least_sq(n_dim, p_deg, num_MC, x_loc);

figure(2)
hold on
box on
histogram(u_vec, 30)
title('Second')
mu_PCE = mean(u_vec);
var_PCE = var(u_vec);

%% x = 0.75
x_loc = 0.75;

u_vec = solver_least_sq(n_dim, p_deg, num_MC, x_loc);

figure(3)
hold on
box on
histogram(u_vec, 30)
title('Third')
mu_PCE = mean(u_vec);
var_PCE = var(u_vec);