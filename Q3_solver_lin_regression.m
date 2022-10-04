
function [u_val] = Q3_solver_lin_regression(x_loc, x_params)
%% Variable definition
    start_x = 0.0;
    end_x = 1.0;
    num_x = 500;           

    source = 5.0;           % s(x) = 5
    rightbc = 1.0;          % u_r = 1
    % x_loc = 0.6;            

%% Initialization
    xgrid = linspace(start_x, end_x, num_x)';
    F = x_params(1);     % Sampling from Gaussian for F(w)
    Y1 = x_params(2) * ones(num_x / 4, 1);
    Y2 = x_params(3) * ones(num_x / 4, 1);
    Y3 = x_params(4) * ones(num_x / 4, 1);
    Y4 = x_params(5) * ones(num_x / 4, 1);
   
    Y = [Y1; Y2; Y3; Y4];           % Initialize for Y
    k = exp(Y);

%% Numerical Solver
    usolution = diffusioneqn(xgrid, F, k, source, rightbc);
    x_coord = abs(xgrid - x_loc) < 0.5 * ((end_x - start_x) / num_x);      % Index of x = 0.6
    
    u_val = usolution(x_coord);
end

