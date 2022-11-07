clear all
clc

x_pos = 0.5;
mu = 1.0;

num_points_n = 20;
num_points_p = 5;

% n_dim = [1, 3, 6, 11, 13, 16, 21];
n_dim = 5;
% n_dim = linspace(1, num_points_n, num_points_n);
% p_deg = linspace(1, num_points_p, num_points_p);
p_deg = [0, 1, 2, 3, 4, 5];
% n_dim = 10;

y_plot = zeros(length(n_dim), length(p_deg));
y_var = zeros(length(n_dim), length(p_deg));

num_samples = 30;

figure(1)
hold on
box on
for i1 = 1:length(n_dim)
    for i2 = 1:length(p_deg)
        k_sample = k_TD_PCE(x_pos, n_dim(i1), p_deg(i2), mu, num_samples);
        y_plot(i1, i2) = abs(mean(k_sample));
        y_var(i1, i2) = abs(var(k_sample));
        i2
    end

end

plot(p_deg, (y_plot), '^ -', 'LineWidth', 2, 'MarkerSize', 8)
plot(p_deg, (y_var), 'v -', 'LineWidth', 2, 'MarkerSize', 8)
plot([0, 5], [exp(1), exp(1)], '-- k' )


xlabel('Number of Polynomial Degree, P')
ylabel('Polynomial Chaos Expansion, k^{*}(x,\omega)')
% ylabel('Error of Polynomial Chaos Expansion, k^{*}(x,\omega) - e')

axis('square')
legend({'Mean of k', 'Variance of k'}, 'Location', 'best')