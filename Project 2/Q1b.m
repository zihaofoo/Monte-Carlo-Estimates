
clear all

%% Variable definition for ODE solver
start_x = 0.0;
end_x = 1.0;
num_x = 200;           

mu_F = -1.0;
sigma_F = sqrt(0.2);
source = 5.0;           % s(x) = 5
rightbc = 1.0;          % u_r = 1

%% Monte Carlo simulation
num_MC = 50;
x_loc = 0.6;

n_dim = 1;
p_deg = 3;
mu = 1.0;
num_sample = 1;

xgrid = linspace(start_x, end_x, num_x)';
F = normrnd(mu_F, sigma_F);     % Sampling from Gaussian for F(w)
k_vec = zeros(num_x, num_MC);

for i1 = 1:num_x
    k_vec(i1, :) = k_TD_PCE(xgrid(i1), n_dim, p_deg, mu, num_MC);
end

u_val = zeros(num_MC, 1);
usolution = zeros(num_x, num_MC);

for i2 = 1:num_MC
    usolution(:, i2) = diffusioneqn(xgrid, F, k_vec(:, i2), source, rightbc);
    % x_coord = abs(xgrid - x_loc) < (0.5 * ((end_x - start_x) / num_x));      % Index of x = x_loc
    % u_val(i2) = usolution(x_coord);
end


