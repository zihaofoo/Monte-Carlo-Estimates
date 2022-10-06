
clear all
clc

%% Original Parameters

mu_F = -2.0;
sigma_F = sqrt(0.5);

mu_Y = -1.0;
sigma_Y = sqrt(1.0);

%% Cross Entropy Optimizer
num_MC = 50;
mu_init = 0;
sigma_init = 4;
opt_max = 100;

theta_j_vec = [mu_init * ones(5,1); sigma_init * ones(5,1)];

theta_0 = [mu_init * ones(5,1); sigma_init * ones(5,1)];
x_mat = zeros(5, num_MC);
theta_mat = zeros(length(theta_j_vec), opt_max);
fun_eval = zeros(opt_max, 1);
Q3c_J_fun(theta_j_vec, x_mat, theta_j_vec)

for i1 = 1:opt_max

    for i2 = 1:num_MC
        x_mat(:, i2) = Q3c_sample_g(theta_j_vec); 
    end

    fun = @(theta_jp1_vec) Q3c_J_fun(theta_jp1_vec, x_mat, theta_j_vec);
    % options_set = optimoptions(@fminunc, 'Algorithm',  'quasi-newton', 'Display','iter', 'FunctionTolerance', 1E-16);
    options_set = optimoptions(@fminunc, 'Algorithm', 'quasi-newton');
    % options_set = optimset('Display', 'final', 'MaxFunEvals', 1E5)
    theta_jp1_params = fminunc(fun, theta_j_vec, options_set); 
    theta_j_vec = theta_jp1_params; 
    theta_mat(:, i1) = theta_jp1_params;

    fun_eval(i1) = Q3c_J_fun(theta_j_vec, x_mat, theta_j_vec);
end

% [M, min_index] = min(fun_eval);
% theta_opt = theta_mat(:, min_index);
theta_opt = theta_jp1_params; 

%% Monte Carlo Solver
u_0 = 40;
x_loc = 0.6;            % Location of random variable
num_MC_solve = 5000;
% theta_opt = [-1.00018467305307; -1.01090152056947; -1.00514799008165; -0.980909765334021; -1.99599491039171; 0.975642591690166; 0.994861764907302; 0.987509066507638; 0.977793306141917; 0.712114814252128];

u_val_vec = zeros(num_MC_solve, 1);
pi_val_vec = zeros(num_MC_solve, 1);
g_val_vec = zeros(num_MC_solve, 1);

num_variance = 100;
p = zeros(num_variance, 1);
for i4 = 1:num_variance
    for i3 = 1:num_MC_solve
        x_vec = Q3c_sample_g(theta_opt);  % Sampling
        pi_val_vec(i3) = Q3c_PDF_pi(x_vec);
        g_val_vec(i3) = Q3c_PDF_g(theta_opt, x_vec);
        u_val_vec(i3) = Q3c_solver(x_loc, x_vec); 
    end

    u_RE_bool = u_val_vec > u_0;
    p(i4) = sum( pi_val_vec(u_RE_bool) ./ g_val_vec(u_RE_bool) ) / length(u_val_vec); 
end
    p_vec = mean(p);
    var_vec = var(p, 1);

error_num = sqrt(num_MC_solve) .* sqrt(var_vec) ./ p_vec
