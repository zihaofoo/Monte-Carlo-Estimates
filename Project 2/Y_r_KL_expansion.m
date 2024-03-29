
function [Y_r, sigma_prime] = Y_r_KL_expansion(x_pos, num_eig, Z_rand)

    if (nargin<2 || isempty(num_eig))
        num_eig = 10;
    end	

    if (nargin<3 || isempty(Z_rand))
        Z_rand = normrnd(0, 1, [num_eig, 1]);
    end	

    mu_Y = 1.0;
    p = 1.0;
    sigma_Y_sq = 0.3;
    L = 0.3;
    params_vec = [sigma_Y_sq, L, p];

    [x_quad, weight_quad] = qrule(num_eig); 
    x_quad = x_quad ./ 2 + 0.5; 
    weight_quad = weight_quad ./ 2;
    
    W_root_mat = diag(sqrt(weight_quad)); 
    C_mat = zeros(length(x_quad), length(x_quad));

    for i1 = 1:length(x_quad)
        C_mat(i1, :) = Cov_func(x_quad(i1), x_quad, params_vec);   % Matrix of covariance
    end

    [phi_mat, lambda_mat] = eig(W_root_mat * C_mat * W_root_mat);
    eigen_vals_vec = diag(lambda_mat);              % Eigenvalues for KL expansion
    eigen_vecs_mat = W_root_mat \ phi_mat;          % Eigenvectors for KL expansion

    eigen_tilde_vec = eigen_vecs_tilde(x_pos, num_eig); 

    Y_r = sum(sqrt(eigen_vals_vec) .* eigen_tilde_vec .* Z_rand) + mu_Y;
    sigma_prime = (sqrt(eigen_vals_vec) .* eigen_tilde_vec)';
    
end