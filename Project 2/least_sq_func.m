
%{
n_dim = 4;
p_deg = 3;
num_MC = 100;
%}

function [error_vec, c_vec] = least_sq_func(n_dim, p_deg, num_MC, x_loc, num_x)
%% Variable definition for ODE solver
start_x = 0.0;
end_x = 1.0;
% num_x = 21;           

mu_F = -1.0;
sigma_F = sqrt(0.2);
source = 5.0;           % s(x) = 5
rightbc = 1.0;          % u_r = 1

if (nargin < 4 || isempty(x_loc))
    x_loc = 0.5;
end
%% Monte Carlo simulation

xgrid = linspace(start_x, end_x, num_x)';
Y_vec = zeros(num_x, num_MC);
Z_rand_MC = zeros(n_dim, num_MC); 
F_MC = zeros(1, num_MC); 

for i1 = 1:num_x
        for i2 = 1:num_MC
            Z_rand = normrnd(0, 1, [n_dim, 1]);
            F = normrnd(mu_F, sigma_F);     % Sampling from Gaussian for F(w)
            
            if i1 == floor(num_x / 2) 
                Z_rand_MC(:, i2) = Z_rand;
                F_MC(:, i2) = (F - mu_F) / sigma_F;
                
            end

            [Y_vec(i1, i2), ~] = Y_r_KL_expansion(xgrid(i1), n_dim, Z_rand);
        
        end
end

k_Y_vec = exp(Y_vec);

%% ODE Solver
u_val_KL = zeros(num_MC, 1);
usol_KL = zeros(num_x, num_MC);

for i2 = 1:num_MC
    usol_KL(:, i2) = diffusioneqn(xgrid, F, k_Y_vec(:, i2), source, rightbc);
    x_coord = abs(xgrid - x_loc) < (0.5 * ((end_x - start_x) / num_x));      % Index of x = x_loc
    u_val_KL(i2) = usol_KL(x_coord, i2);
end

%% Least-square Approximation
tomil = TotalOrderMultiIndexLattice(n_dim + 1, p_deg);
tomil.init();
Mtotalorder = tomil.get_midx_matrix();

num_PCE_terms = size(Mtotalorder, 1)
z_vec = vertcat(Z_rand_MC, F_MC);
v_mat = zeros(num_MC, num_PCE_terms);

for i2 = 1:num_MC
    for i1 = 1:num_PCE_terms
        alpha_vec = hermite_poly(z_vec(:, i2), Mtotalorder(i1, :));
        % normal_vec = factorial(Mtotalorder(i1, :)');
        % v_mat(i2, i1) = prod(alpha_vec ./ normal_vec);
        v_mat(i2, i1) = prod(alpha_vec);
    end
end

G_mat = v_mat' * v_mat;
f_prime_vec = v_mat' * u_val_KL; 
c_vec = linsolve(G_mat, f_prime_vec);           % Coefficient vectors

error_vec = norm(v_mat * c_vec - u_val_KL, 2);

end
