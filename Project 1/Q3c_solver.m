
function [u_val] = Q3c_solver(x_loc, x_vec)
%% Variable definition
    start_x = 0.0;
    end_x = 1.0;
    num_x = 160;           

    source = 5.0;           % s(x) = 5
    rightbc = 1.0;          % u_r = 1
    % x_loc = 0.6;            

%% Initialization
    xgrid = linspace(start_x, end_x, num_x)';
    Y1 = x_vec(1) * ones(num_x / 4, 1);
    Y2 = x_vec(2) * ones(num_x / 4, 1);
    Y3 = x_vec(3) * ones(num_x / 4, 1);
    Y4 = x_vec(4) * ones(num_x / 4, 1);
    F = x_vec(5);     % Sampling from Gaussian for F(w)

    Y = [Y1; Y2; Y3; Y4];           % Initialize for Y
    k = exp(Y);

%% Numerical Solver
    usolution = diffusioneqn(xgrid, F, k, source, rightbc);
    x_coord = abs(xgrid - x_loc) < 0.5 * ((end_x - start_x) / num_x);      % Index of x = 0.6
    
    u_val = usolution(x_coord);
end