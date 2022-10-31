

function [output] = Q3c_J_fun(theta_vec, x_mat, theta_j_vec)
    
    [n_rows, n_cols] = size(x_mat);
    J_vec = zeros(n_cols, 1);
    x_loc = 0.6; 
    u_0 = 40;
    u_val_vec = zeros(n_cols, 1);
    for i1 = 1:n_cols
        pi_x = Q3c_PDF_pi(x_mat(:,i1));
        g_x_theta_j = Q3c_PDF_g(theta_j_vec, x_mat(:,i1));
        log_g = log(Q3c_PDF_g(theta_vec, x_mat(:,i1))); 
        
        u_val_vec = Q3c_solver(x_loc, x_mat(:,i1)); 
        
        J_vec(i1) = pi_x / g_x_theta_j * log_g; 

    end
    
    output = - sum(u_val_vec .* J_vec) / length(J_vec);

end