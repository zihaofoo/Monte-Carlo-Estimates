
function k_sample = k_TD_PCE(x_pos, n_dim, p_deg, mu, num_sample)
    
    % n_dim = number of stochastic dimension
    % p_deg = total polynomial degree 
    % num_sample = number of independent samples at a given x_pos
    
    tomil = TotalOrderMultiIndexLattice(n_dim, p_deg);
    tomil.init();
    Mtotalorder = tomil.get_midx_matrix();
    
    k_sample = zeros(num_sample, 1);
    
    for j1 = 1:num_sample
        num_PCE_terms = size(Mtotalorder, 1); 
        Z_rand = normrnd(0, 1, [n_dim, 1]);   % Sample from standard normal
        [~, sigma_prime] = Y_r_KL_expansion(x_pos, n_dim, Z_rand); % Computes eigenvals and vectors
        sigma_prime = real(sigma_prime);

        k_PCE = zeros(num_PCE_terms, 1);        
        % C_alpha_save = zeros(num_PCE_terms, 1);

        for i1 = 1:num_PCE_terms
            C_alpha_vec = C_alpha(mu, sigma_prime, Mtotalorder(i1, :));
            % C_alpha_save(i1) = C_alpha(mu, sigma_prime, Mtotalorder(i1, :));
            He_vec = hermite_poly(Z_rand, Mtotalorder(i1, :));
            k_PCE(i1) = C_alpha_vec * prod(He_vec); 
        end
        k_sample(j1) = sum(k_PCE); 
    end
end