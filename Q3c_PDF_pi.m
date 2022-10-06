

function [output] = Q3c_PDF_pi(x_vec)
    

    mu_F = -2.0;
    sigma_F = sqrt(0.5);

    mu_Y = -1.0;
    sigma_Y = sqrt(1.0);

    Y1 = normpdf(x_vec(1), mu_Y, sigma_Y);
    Y2 = normpdf(x_vec(2), mu_Y, sigma_Y);
    Y3 = normpdf(x_vec(3), mu_Y, sigma_Y);
    Y4 = normpdf(x_vec(4), mu_Y, sigma_Y);
    F = normpdf(x_vec(5), mu_F, sigma_F);     % Sampling from Gaussian for F(w)

    output = Y1 * Y2 * Y3 * Y4 * F;
end