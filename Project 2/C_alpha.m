

function C_out = C_alpha(mu, sigma_prime, alpha_vec)

    n_dim = length(sigma_prime);
    
    numer = exp(mu);
    denom = 0;
    for i1 = 1:n_dim
        numer = numer * exp(0.5 * sigma_prime(i1)^2) * (sigma_prime(i1) ^ alpha_vec(i1)); 
        denom = denom + factorial(alpha_vec(i1));
    end
    
    C_out = numer / denom;

end