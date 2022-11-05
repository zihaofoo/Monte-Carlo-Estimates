

function [output] = hermite_poly(x_pos, num_dim)

    He_vec = zeros(num_dim + 1, 1);
    He_vec(1) = 1;
    
    for i1 = 2:(num_dim + 1)
        % He_vec(i1) =  hermiteH(i1 - 1, (x_pos / sqrt(2)));
        He_vec(i1) = (2 ^ (-(i1 - 1)/2)) * hermiteH(i1 - 1, (x_pos / sqrt(2)));
    end

    output = He_vec;
end