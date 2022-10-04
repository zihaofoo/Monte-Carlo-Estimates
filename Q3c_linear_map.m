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

num_MC = 1E+4; 

F = zeros(num_MC, 1); 
Y1 = zeros(num_MC, 1); 
Y2 = zeros(num_MC, 1); 
Y3 = zeros(num_MC, 1); 
Y4 = zeros(num_MC, 1); 
u_val_vec = zeros(num_MC, 1); 
u_0 = 40;

%% Monte Carlo Estimator

for i1 = 1:num_MC
    F(i1) = normrnd(mu_F, sigma_F);     % Sampling from Gaussian for F(w)
    Y1(i1) = normrnd(mu_Y, sigma_Y, 1);
    Y2(i1) = normrnd(mu_Y, sigma_Y, 1);
    Y3(i1) = normrnd(mu_Y, sigma_Y, 1);
    Y4(i1) = normrnd(mu_Y, sigma_Y, 1);
    u_val_vec(i1) = Q3_solver_lin_regression(x_loc, [F(i1), Y1(i1), Y2(i1), Y3(i1), Y4(i1)]);
end
        
%% Regression of Linear Map
x0 = [1, 1, 1, 1, 1, 1]; 
fun = @(x_vec) lin_regression_sub(x_vec, Y1, Y2, Y3, Y4, F, u_val_vec);
options_set = optimoptions(@fsolve, 'Algorithm', 'trust-region', 'Display','iter', 'FunctionTolerance', 1E-16);
x_params = fsolve(fun, x0, options_set) 

u_est = solver_linear_map(x_params, Y1, Y2, Y3, Y4, F);

u_err = mean(abs(u_est - u_val_vec))

%{
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
%}