clear all

% x = linspace(0, 1, 100);
num_eig = 100;
x = 0.5;

eig_val = zeros(length(x), num_eig);

num_sample = 100;
Y_r = zeros(num_sample, 1);


for i2 = 1:num_sample
    for i1 = 1:length(x)
        eig_val (i1, :) = eigen_vecs_tilde(x(i1), num_eig);
        [Y_r(i2), sigma_prime] = Y_r_KL_expansion(x(i1), num_eig);
    end
    i2
end


figure(1)

hold on
box on
plot(x, Y_r)


