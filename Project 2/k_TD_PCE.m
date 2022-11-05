
function k_sample = k_TD_PCE(x_pos, n_dim, p_deg, mu, num_sample)
    
    % n_dim = number of stochastic dimension
    % p_deg = total polynomial degree 
    
    tomil = TotalOrderMultiIndexLattice(n_dim, p_deg);
    tomil.init();
    Mtotalorder = tomil.get_midx_matrix();
    
    k_sample = zeros(num_sample, 1);

    for j1 = 1:num_sample
        num_PCE_terms = size(Mtotalorder, 1); 
        [~, sigma_prime] = Y_r_KL_expansion(x_pos, n_dim); 
        sigma_prime = real(sigma_prime);

        k_PCE = 0;
        He_vec = hermite_poly(x_pos, p_deg);

        for i1 = 1:num_PCE_terms
            C_alpha_vec = C_alpha(mu, sigma_prime, Mtotalorder(i1, :));
            He_alpha = prod(He_vec(Mtotalorder(i1, :) + 1));
            k_PCE = k_PCE + (C_alpha_vec * He_alpha); 
        end
    
        k_sample(j1) = k_PCE; 
    end
end