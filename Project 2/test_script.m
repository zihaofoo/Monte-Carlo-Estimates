clear all
clc

n_dim = 20;
p_deg = 2;
mu_Y = 1;
x_pos = 0.5;
sigma_sq = 0.3;
num_sample = 1;
y_sample = zeros(num_sample, 1);
 
var_analytical = (exp(sigma_sq) - 1) * exp(2 * mu_Y + sigma_sq);

%{
for i1 = 1:num_sample
    y_sample(i1) = Y_r_KL_expansion(x_pos, n_dim);    % Testing KL expansion of Y
end

k_KL_vec = exp(y_sample); 

%}
%
k_sample = k_TD_PCE(x_pos, n_dim, p_deg, mu_Y, num_sample); 

mu_PCE = mean(k_sample)
var_PCE = var(k_sample)

%}

%

%{
tomil = TotalOrderMultiIndexLattice(n_dim, p_deg);
tomil.init();
Mtotalorder = tomil.get_midx_matrix();
num_PCE_terms = size(Mtotalorder, 1); 
Z_rand = normrnd(0, 1, [n_dim, 1]);   % Sample from standard normal

He_mat = zeros(num_PCE_terms, n_dim);

for i1 = 1:num_PCE_terms
    alpha_vec = Mtotalorder(i1, :);
    He_mat(i1, :) = hermite_poly(Z_rand, alpha_vec);
end


%}