

function [output] = Q3c_sample_pi(theta_vec)
    

    mu_F = -2.0;
    sigma_F = sqrt(0.5);

    mu_Y = -1.0;
    sigma_Y = sqrt(1.0);

    Y1 = normrnd(mu_Y, sigma_Y, 1);
    Y2 = normrnd(mu_Y, sigma_Y, 1);
    Y3 = normrnd(mu_Y, sigma_Y, 1);
    Y4 = normrnd(mu_Y, sigma_Y, 1);
    F = normrnd(mu_F, sigma_F);     % Sampling from Gaussian for F(w)

    output = [Y1; Y2; Y3; Y4; F];

end