function [usolution] = diffusioneqn(xgrid, F, k, source, rightbc)
% diffusion.m: 
%
% Solve 1-D diffusion equation with given diffusivity field k
% and left-hand flux F.
%
% ARGUMENTS: 
%     xgrid = vector with grid points
%         F = flux at left-hand boundary, k*du/dx = -F 
%    source = source term, either a vector of values at points in xgrid
%             or a constant
%   rightbc = Dirichlet BC on right-hand boundary
%
% Domain is given by xgrid (should be [0,1])
%

N = length(xgrid);
h = xgrid(N)-xgrid(N-1); % assuming uniform grid

% Set up discrete system f = Au + b using second-order FD
A = sparse(N-1,N-1);
b = zeros(N-1,1);
if (isscalar(source))
    f = -source * ones(N-1,1);
else
    f = -source(1:N-1);
end


% diagonal entries
A = A - 2*diag(k(1:N-1)) - diag(k(2:N)) - diag([k(1); k(1:N-2)]);

% superdiagonal
A = A + diag(k(1:N-2),1)  + diag(k(2:N-1),1);

% subdiagonal
A = A + diag(k(1:N-2),-1) + diag(k(2:N-1),-1);

A = A / (2 * h^2);

% Treat Neumann BC on left side
A(1,2) = A(1,2) + k(1) / (h^2);
b(1) = 2*F/h;

% Treat Dirichlet BC on right side
b(N-1) = rightbc * (k(N) + k(N-1)) / (2 * h^2);

% Solve it: Au = f-b
uinternal = A \ (f - b);

usolution = [uinternal; rightbc];

end

