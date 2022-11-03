
function output = Y_r_KL_expansion(x_pos, num_eig)

    if (nargin<2 || isempty(num_eig))
        num_eig = 10;
    end	

    mu_Y = 1.0;
    p = 1.0;
    sigma_Y_sq = 0.3;
    L = 0.3;
    params_vec = [sigma_Y_sq, L, p];


    [x_quad, weight_quad] = qrule(num_eig); 
    W_root_mat = diag(sqrt(weight_quad)); 
    C_mat = zeros(length(x_quad), length(x_quad));

    for i1 = 1:length(x_quad)
        for i2 = 1:length(x_quad)
            C_mat(i1, i2) = Cov_func(x_quad(i1), x_quad(i2), params_vec);   % Matrix of covariance
        end    
    end

    [phi_mat, lambda_mat] = eig(W_root_mat * C_mat * W_root_mat);
    eigen_vals_vec = diag(lambda_mat);              % Eigenvalues for KL expansion
    eigen_vecs_mat = W_root_mat \ phi_mat;     % Eigenvectors for KL expansion


    Y_r = 0 + mu_Y;
    Z_rand = normrnd(0, 1, num_eig, 1);
    eigen_tilde_vec = eigen_vecs_tilde(x_pos, num_eig); 

    for i1 = 1:num_eig
        Y_r = Y_r + (sqrt(eigen_vals_vec(i1)) * eigen_tilde_vec(i1) * Z_rand(i1)); 
    end

    output = Y_r;
end