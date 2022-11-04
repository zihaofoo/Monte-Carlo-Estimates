clear all
clc

x_pos = 0.5;
mu = 1.0;

num_points_n = 3;
num_points_p = 4;

n_dim = linspace(1, num_points_n, num_points_n);
p_deg = linspace(1, num_points_p, num_points_p);
p_deg = 5;

figure(1)
hold on
box on

y_plot = zeros(length(n_dim), length(p_deg));

for i1 = 1:length(n_dim)
    for i2 = 1:length(p_deg)
        y_plot(i1, i2) = k_TD_PCE(x_pos, n_dim(i1), p_deg(i2), mu);
    end
    
    plot(p_deg, (y_plot(i1, :) / y_plot(i1, end)), '^ -', 'LineWidth', 2, 'MarkerSize', 8) 
    % plot(n_dim, (y_plot(i1, :)), '^ -', 'LineWidth', 2, 'MarkerSize', 8) 
    i1
   
end

xlabel('Number of Polynomial Degree, P')
ylabel('Normalized Polynomial Chaos Expansion, k^{*}(x,\omega)')
axis('square')

legend(string(n_dim), 'Location', 'best')