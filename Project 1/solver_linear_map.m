

function [u_est] = solver_linear_map(x_params, Y1, Y2, Y3, Y4, F)
    
    u_est = x_params(1) + x_params(2) .* Y1 + x_params(3) .* Y2 + x_params(4) .* Y3 + x_params(5) .* Y4 + x_params(6) .* F; 

end