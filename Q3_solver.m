
function [u_val] = Q3_solver(x_loc)
%% Variable definition
    start_x = 0.0;
    end_x = 1.0;
    num_x = 1000;           

    mu_F = -2.0;
    sigma_F = sqrt(0.5);

    mu_Y = -1.0;
    sigma_Y = sqrt(1.0);

    source = 5.0;           % s(x) = 5
    rightbc = 1.0;          % u_r = 1
    % x_loc = 0.6;            

%% Initialization
    xgrid = linspace(start_x, end_x, num_x)';
    F = normrnd(mu_F, sigma_F);     % Sampling from Gaussian for F(w)
    Y1 = normrnd(mu_Y, sigma_Y, num_x / 4, 1);
    Y2 = normrnd(mu_Y, sigma_Y, num_x / 4, 1);
    Y3 = normrnd(mu_Y, sigma_Y, num_x / 4, 1);
    Y4 = normrnd(mu_Y, sigma_Y, num_x / 4, 1);
    Y = [Y1; Y2; Y3; Y4];           % Initialize for Y
    k = exp(Y);

%% Numerical Solver
    usolution = diffusioneqn(xgrid, F, k, source, rightbc);
    x_coord = abs(xgrid - x_loc) < 0.5 * ((end_x - start_x) / num_x);      % Index of x = 0.6
    
    u_val = usolution(x_coord);
end

