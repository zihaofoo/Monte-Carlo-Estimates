clear all

x_pos = 0.5; 
num_eig = 50;
num_sample = 10000;

y_sample = zeros(num_sample, 1); 

for i1 = 1:num_sample
    y_sample(i1) = Y_r_KL_expansion(x_pos, num_eig);
end

mean(y_sample)
var(y_sample)