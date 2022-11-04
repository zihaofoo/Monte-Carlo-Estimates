clear all 
% Plotting script for Question 1(a)

x_pos = 0.5;
num_eig = 10000;

[Y_r, eigen_vals_vec] = Y_r_KL_expansion_old(x_pos, num_eig); 
eigen_vals_vec = real(eigen_vals_vec);

n = [5; 7; 10; 50; 70; 100; 500; 700; 1000; 7000; 10000]; 
error_vec = zeros(length(n), 1);

for i1 = 1:length(n)
    error_vec(i1) = sum(eigen_vals_vec(n(i1):end - 1));
end
figure(1)
hold on
box on

plot(n, error_vec, '^ - k')
ylabel('Error in Truncation, \epsilon (-)');
xlabel('Stochastic dimension, n (-) ');
axis('square')
