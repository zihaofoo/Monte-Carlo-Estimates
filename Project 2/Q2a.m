
clear all 

%% Variable definition for ODE solver
start_x = 0.0;
end_x = 1.0;
num_x = 21;           

mu_F = -1.0;
sigma_F = sqrt(0.2);
source = 5.0;           % s(x) = 5
rightbc = 1.0;          % u_r = 1

%% Monte Carlo simulation for P-deg
num_MC = 250;
n_dim = 4;
p_deg = linspace(1, 5, 5);
error_vec = zeros(length(p_deg), 1);

for i1 = 1:length(error_vec)
    [error_vec(i1), c_vec] = least_sq_func(n_dim, p_deg(i1), num_MC);
end

figure(1)
plot(p_deg, error_vec/ num_MC)
xlabel('Polynomial degree, p')
ylabel('L2 error of ||Vc - f||')
axis('square')

%% Monte Carlo simulation for num_MC
num_MC = 70:50:270;
n_dim = 4;
p_deg = 3;
mu = 1.0;
error_vec = zeros(length(num_MC), 1);

for i1 = 1:length(error_vec)
    [error_val, c_vec] = least_sq_func(n_dim, p_deg, num_MC(i1));
    error_vec(i1) = error_val / num_MC(i1);
end

figure(2)
plot(num_MC, error_vec)
xlabel('Number of Monte Carlo simulations, s')
ylabel('L2 error of ||Vc - f||')
axis('square')



