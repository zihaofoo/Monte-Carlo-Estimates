

function [output] = hermite_poly(x_pos, num_dim)

    He_vec = zeros(num_dim, 1);
    
    He_0 = @(x)(1);
    He_1 = @(x)(x);
    He_2 = @(x)(x.^2 - 1);
    He_3 = @(x)(x.^3 - 3 .* x);
    He_4 = @(x)(x.^4 - 6 .* x.^2 + 3);
    He_5 = @(x)(x.^5 - 10 .* x.^3 + 15 .* x);
    He_6 = @(x)(x.^6 - 15 .* x.^4 + 45 .* x.^2 - 15);
    He_7 = @(x)(x.^7 - 21 .* x.^5 + 105 .* x.^3 - 105 .* x);
    He_8 = @(x)(x.^8 - 28 .* x.^6 + 210 .* x.^4 - 420 .* x.^2 + 105); 
    He_9 = @(x)(x.^9 - 36 .* x.^7 + 378 .* x.^5 - 1260 .* x.^3 + 945 .* x); 
    He_10 = @(x)(x.^10 - 45 .* x.^8 + 630 .* x.^6 - 3150 .* x.^4 + 4725 .* x .^2 - 945); 
    
    He_func = {He_0, He_1, He_2, He_3, He_4, He_5 ,He_6, He_7, He_8, He_9, He_10};  

    for i1 = 1:num_dim
        He_vec(i1) = He_func{i1}(x_pos);    
    end

    output = He_vec;
end