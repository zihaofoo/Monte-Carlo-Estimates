
function [output] = solver_least_sq(n_dim, p_deg, num_MC, x_loc, num_x)
    
    [error_vec, c_vec] = least_sq_func(n_dim, p_deg, num_MC, x_loc, num_x);
                
    Z_rand_MC = zeros(n_dim, num_MC); 
    F_MC = zeros(1, num_MC); 
    
    % num_x = 21;
    mu_F = -1.0;
    sigma_F = sqrt(0.2);

    for i1 = 1:num_x
        for i2 = 1:num_MC
            Z_rand = normrnd(0, 1, [n_dim, 1]);
            F = normrnd(mu_F, sigma_F);     % Sampling from Gaussian for F(w)
            
            if i1 == floor(num_x / 2) 
                Z_rand_MC(:, i2) = Z_rand;
                F_MC(:, i2) = (F - mu_F) / sigma_F;
            end
        
        end
    end

    tomil = TotalOrderMultiIndexLattice(n_dim + 1, p_deg);
    tomil.init();
    Mtotalorder = tomil.get_midx_matrix();

    num_PCE_terms = size(Mtotalorder, 1);
    z_vec = vertcat(Z_rand_MC, F_MC);
    v_mat = zeros(num_MC, num_PCE_terms);

    for i2 = 1:num_MC
        for i1 = 1:num_PCE_terms
            alpha_vec = hermite_poly(z_vec(:, i2), Mtotalorder(i1, :));
            v_mat(i2, i1) = prod(alpha_vec);
        end
    end
    
    u_vec = v_mat * c_vec;

    output = u_vec;

end