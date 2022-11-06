

function [output] = hermite_poly(Z_rand, alpha_vec)
    % Z_rand = (n,1) vector of random variable
    % alpha_vec = (n,1) vector of polynomial orders of the corresponding RV
    
    He_vec = zeros(length(alpha_vec), 1);

    for i1 = 1:length(alpha_vec)
        if alpha_vec(i1) == 0
            He_vec(i1) = 1;
        else
            He_vec(i1) = (2 .^ (- (alpha_vec(i1))/2)) * hermiteH(alpha_vec(i1), (Z_rand(i1) ./ sqrt(2)));
        end
    end

    output = He_vec;

    %{
    He_vec = zeros(num_dim + 1, 1);
    He_vec(1) = 1;
    
    for i1 = 2:(num_dim + 1)
        % He_vec(i1) =  hermiteH(i1 - 1, (x_pos / sqrt(2)));
        He_vec(i1) = (2 ^ (-(i1 - 1)/2)) * hermiteH(i1 - 1, (z_pos / sqrt(2)));
    end

    output = He_vec;
    %}
end