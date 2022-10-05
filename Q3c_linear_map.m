%% Solution Script to Question 3c 
clear all 
clc

%% Variable definition
start_x = 0.0;
end_x = 1.0;
num_x = 200;           

mu_F = -2.0;
sigma_F = sqrt(0.5);

mu_Y = -1.0;
sigma_Y = sqrt(1.0);

source = 5.0;           % s(x) = 5
rightbc = 1.0;          % u_r = 1
x_loc = 0.6;            % Location of random variable

num_MC = 1E+3; 

F = zeros(num_MC, 1); 
Y1 = zeros(num_MC, 1); 
Y2 = zeros(num_MC, 1); 
Y3 = zeros(num_MC, 1); 
Y4 = zeros(num_MC, 1); 
u_val_vec = zeros(num_MC, 1); 


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
% x_params = fsolve(fun, x0, options_set); 

scal_constant = 10;
x_params = [2.0637, -0.0807, -0.0683, -0.0654, -0.0610, -1.1237];
W = solver_linear_map(x_params, Y1, Y2, Y3, Y4, F);

u_err = (mean(W) - mean(u_val_vec)) /  mean(u_val_vec);
mu_W = x_params(1) + (x_params(2) * mu_Y) + (x_params(3) * mu_Y) + (x_params(4) * mu_Y) + (x_params(5) * mu_Y) + (x_params(6) * mu_F); 
%% Control Variate 

cov_U_W = mean(u_val_vec .* W) - (mean(u_val_vec) * mean(W));
c_ast = - cov_U_W / var(W);  
Z = W + c_ast .* (W - mu_W); 

mean(Z)
var(u_val_vec)
var(Z)

