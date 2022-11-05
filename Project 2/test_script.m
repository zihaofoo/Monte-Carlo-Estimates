clear all
clc

x_pos = 1; 
num_eig = 2;
num_sample = 100;
p_deg = 5;
mu_Y = 1;

y_sample = zeros(num_sample, 1); 
k_sample = zeros(num_sample, 1); 


for i1 = 1:num_sample
    y_sample(i1) = Y_r_KL_expansion(x_pos, num_eig);    % Testing KL expansion of Y
    i1
end

k_sample = k_TD_PCE(x_pos, num_eig, p_deg, mu_Y, num_sample);


mean(y_sample)
mean(k_sample)

