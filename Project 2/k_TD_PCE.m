
function output = k_TD_PCE(x_pos, n_dim, p_deg, mu)
    
    % n_dim = number of stochastic dimension
    % p_deg = total polynomial degree 

    tomil = TotalOrderMultiIndexLattice(n_dim, p_deg);
    tomil.init();
    Mtotalorder = tomil.get_midx_matrix();   
    num_PCE_terms = size(Mtotalorder, 1);
    [~, sigma_prime] = Y_r_KL_expansion(x_pos, n_dim); 
    
    k_PCE = 0;
    for i1 = 1:num_PCE_terms
       C_alpha_vec = C_alpha(mu, sigma_prime, Mtotalorder(i1, :));  
       He_alpha = 1; 
       
       for i2 = 1:n_dim
           He_alpha = He_alpha * hermite_poly(x_pos, Mtotalorder(i1, i2));
       end
       k_PCE = k_PCE + C_alpha_vec * He_alpha; 
    end
    
    output = k_PCE;
end