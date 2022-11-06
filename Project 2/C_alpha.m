

function C_out = C_alpha(mu, sigma_prime, alpha_vec)

   numer = exp(mu) * prod(exp(0.5 .* sigma_prime .^ 2)) * prod(sigma_prime .^ alpha_vec); 
   denom = prod(factorial(alpha_vec));

   C_out = numer / denom;

end