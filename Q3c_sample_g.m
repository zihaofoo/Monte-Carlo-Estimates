

function [output] = Q3c_sample_g(theta_vec)
    
    mu_Y1 = theta_vec(1);
    mu_Y2 = theta_vec(2); 
    mu_Y3 = theta_vec(3); 
    mu_Y4 = theta_vec(4); 
    mu_F = theta_vec(5); 
    sigma = theta_vec(6);


    Y1 = normrnd(mu_Y1, sigma, 1);
    Y2 = normrnd(mu_Y2, sigma, 1);
    Y3 = normrnd(mu_Y3, sigma, 1);
    Y4 = normrnd(mu_Y4, sigma, 1);
    F = normrnd(mu_F, sigma);     % Sampling from Gaussian for F(w)

    output = [Y1; Y2; Y3; Y4; F];

end