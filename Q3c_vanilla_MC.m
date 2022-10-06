%% Solution Script to Question 3c
clear all 
clc

%% Variable definition
start_x = 0.0;
end_x = 1.0;
num_x = 1000;           

mu_F = -2.0;
sigma_F = sqrt(0.5);

mu_Y = -1.0;
sigma_Y = sqrt(1.0);

source = 5.0;           % s(x) = 5
rightbc = 1.0;          % u_r = 1
x_loc = 0.6;            % Location of random variable

num_MC = [1E+1, 1E+2, 1E+3, 1E+4, 1E+5];
num_MC = 5E+3;
% u_val_vec = zeros(num_MC, 1);
h_n = zeros(length(num_MC), 1);
h_n_sq = zeros(length(num_MC), 1);
sigma_u = zeros(length(num_MC), 1);
sigma_u_sq = zeros(length(num_MC), 1);
del_x_confi = zeros(length(num_MC), 1);
var_val_vec = zeros(length(num_MC), 1);

u_0 = 40;
num_variance = 100;
p_vec = zeros(length(num_MC), 1);
var_vec = zeros(length(num_MC), 1);

%% Monte Carlo Estimator

for i3 = 1:length(num_MC)
    p = zeros(num_variance, 1);

    for i2 = 1:num_variance

        u_val_vec = zeros(num_MC(i3), 1);

        for i1 = 1:num_MC(i3)
            u_val_vec(i1) = Q3_solver(x_loc);
        end
        
        u_RE_bool = u_val_vec > u_0;
        p(i2) = sum(u_RE_bool)/ num_MC(i3);

    end
    p_vec(i3) = mean(p);
    var_vec(i3) = var(p, 1);
end

error_num = sqrt(num_MC') .* sqrt(var_vec) ./ p_vec

figure(1)
plot(num_MC, p_vec, 'k - o', 'LineWidth', 2)
ylabel('Expected value p from MC simulation', 'FontSize', 18)

yyaxis right
plot(num_MC, sqrt(var_vec), 'r - o', 'LineWidth', 2)
xlabel('Number of sample points in MC simulation', 'FontSize', 18)
ylabel('Standard error of p from MC simulation', 'FontSize', 18)
axis('square')
% Expectation of u(x=0.6)
% figure(1)
% plot((num_MC), (h_n))
% axis('square')
% xlabel('Number of sample points in MC simulation', 'FontSize', 18)
% ylabel('Expected value from MC simulation', 'FontSize', 18)

% Variance of u(x=0.6)
% figure(2)
% plot((num_MC), (var_val_vec), 'k- ^', 'LineWidth', 2)
% axis('square')
% xlabel('Number of sample points in MC simulation', 'FontSize', 18)
% ylabel('Expected variance from MC simulation', 'FontSize', 18)
