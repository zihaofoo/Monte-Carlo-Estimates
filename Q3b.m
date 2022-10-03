%% Solution Script to Question 3b
clear all 
clc

%% Variable definition
start_x = 0.0;
end_x = 1.0;
num_x = 100;           

mu_F = -2.0;
sigma_F = sqrt(0.5);

mu_Y = -1.0;
sigma_Y = sqrt(1.0);

source = 5.0;           % s(x) = 5
rightbc = 1.0;          % u_r = 1
x_loc = 0.6;            % Location of random variable

num_MC = [50, 100, 500, 1000, 5000, 10000, 50000];
% u_val_vec = zeros(num_MC, 1);
h_n = zeros(length(num_MC), 1);
h_n_sq = zeros(length(num_MC), 1);
sigma_u = zeros(length(num_MC), 1);
sigma_u_sq = zeros(length(num_MC), 1);
del_x_confi = zeros(length(num_MC), 1);
var_val_vec = zeros(length(num_MC), 1);

%% Monte Carlo Estimator

for i2 = 1:length(num_MC)

    u_val_vec = zeros(num_MC(i2), 1);

    for i1 = 1:num_MC(i2)
        u_val_vec(i1) = Q3_solver(x_loc);
    end
    
    % figure(1)
    % histogram(u_val_vec, 1000)
    % set(gca,'FontSize', 20)
    % axis('square')
    % xlabel('u (x = 0.6, \omega)', 'FontSize', 18)
    % ylabel('Frequency of MC estimate', 'FontSize', 18)

    h_n(i2) = sum(u_val_vec) / length(u_val_vec);
    h_n_sq(i2) = sum(u_val_vec .^2) / length(u_val_vec);

    var_val_vec(i2) = sum( (u_val_vec - h_n(i2)).^2 ) / length(u_val_vec);

    sigma_u(i2) = sqrt(var(u_val_vec, 0));
    sigma_u_sq(i2) = sqrt(var((u_val_vec - h_n(i2)).^2, 0));

    del_x_confi(i2) = h_n(i2) + (sigma_u(i2) * 1.62 / sqrt(num_MC(i2)) );
end
var_n = h_n_sq - h_n.^2; 

% Expectation of u(x=0.6)
figure(1)
plot((num_MC), (h_n))
axis('square')
xlabel('Number of sample points in MC simulation', 'FontSize', 18)
ylabel('Expected value from MC simulation', 'FontSize', 18)

% Variance of u(x=0.6)
figure(2)
plot((num_MC), (var_val_vec), 'k- ^', 'LineWidth', 2)
axis('square')
xlabel('Number of sample points in MC simulation', 'FontSize', 18)
ylabel('Expected variance from MC simulation', 'FontSize', 18)
