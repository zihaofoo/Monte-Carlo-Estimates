
function [output] = Cov_func(x1, x2, params)

  if nargin < 3
    sigma_sq = 0.3;
    L = 0.3;
    p = 1.0;

  else
    sigma_sq = params(1);
    L = params(2);
    p = params(3); 
  end

  output = sigma_sq * exp(-(1 / p) .* (abs(x1 - x2) ./ L) .^ p);

end