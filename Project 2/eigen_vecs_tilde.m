
function output = eigen_vecs_tilde(x_pos, num_eig)
    % Computes the eigenvectors at non-quadrature points


    % Initialization
    if (nargin<2 || isempty(num_eig))
        num_eig = 10;
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
         C_mat(i1, :) = Cov_func(x_quad(:), x_quad(i1), params_vec);   % Matrix of covariance
    end

    [phi_mat, lambda_mat] = eig(W_root_mat * C_mat * W_root_mat);
    eigen_vals_vec = diag(lambda_mat);              % Eigenvalues for KL expansion
    eigen_vecs_mat = W_root_mat \ phi_mat;          % Eigenvectors for KL expansion

    psi_tilde_vec = zeros(num_eig, 1);

    for i3 = 1:length(x_quad)   % i3 = i in psi_i

        tilde_sum_i = 0;

        for i4 = 1:length(x_quad)
            tilde_sum_i = tilde_sum_i + (Cov_func(x_pos, x_quad(i4), params_vec) * weight_quad(i4) * eigen_vecs_mat(i4, i3)); 
        end
        
        psi_tilde_vec(i3) = tilde_sum_i / eigen_vals_vec(i3);   % Divide sum by the eigenvalue.
    end
    
    output = (psi_tilde_vec); 
end
