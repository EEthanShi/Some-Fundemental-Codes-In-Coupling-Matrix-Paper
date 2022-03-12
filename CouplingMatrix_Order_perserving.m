function [T, info] = CouplingMatrix_Order_perserving(C, n, m, p, q, lambda1,lambda2, D,P,solver, options)
% This function is to define and solve the OT problem by using the optimization
% on the coupling matrix manifolds.
% The OT objective function is the version defined in 
%
%   Marco Cuturi, Sinkhorn distances: Lightspeed computation of optimal transport,
%   2013 NIPS, pp 2292 - 2300.
% 
% The usage
%    [T, info] = CouplingMatrix_OT(C, n, m, p, q, lambda, opts)
%
% Inputs:
% C:  The positive cost matrix of size nxm
% n and m:  number of sources and number of targets for transport
% p:  The positive source allocation/distribution of size n
% q:  The positive target allocation of size m
%     if p or q are missing, its default values are 1/n*ones(n,1) and
%     1/m*ones(m,1)
% lambda: the regulasizer in OT objective, default = 200
% solver:  a string 'CG | SD | RTR '
%          CG: Conjugate Gradient Descent 
%          SD:  Steep Gradient Descent with line search:  Not stable
%          RTR: Riemannian Trust Region (second order)
% opts: The structure provided to ManOpt solver
%
% Outputs:
% T:  The transport matrix, the minimiser of the objective
% info:  A data structure from the ManOpt solver, please refer to
%        ManOpt.org
%
% 

if ~exist('p', 'var') || isempty(p)
    p = ones(n,1) / n;
end
    
if ~exist('q', 'var') || isempty(q)
    q = ones(m,1) / m;
end

%if abs(1-sum(p)/sum(q)) > eps*10
%    error('p and q are not coupled, sum(p) equal to sum(q), please!')
%end
    
if ~exist('lambda1', 'var') || isempty(lambda1)
    lambda1 = 50;
end

if ~exist('lambda2', 'var') || isempty(lambda2)
    lambda2 = 0.1;
end

 

if ~exist('solver', 'var') || isempty(solver)
    solver = 'CG';
end

if ~exist('options', 'var') || isempty(options)
    options.checkperiod = 1;
    options.maxiter = 100;
    options.verbosity = 0;
end

%% Create the problem structure for the coupling matrix
% n \times m matrices with column rum to p and row sums to q.
% a space of matrices in dimension (n-1)(m-1).

problem.M = couplingmatrixfactory(n, m, p, q); 
problem.cost = @cost;
problem.egrad = @egrad;
problem.ehess = @ehess;
%% Solve

if strcmp(solver, 'CG')
   % Minimize the cost function using the Conjugate Gradients algorithm.
   [T, ~, info] = conjugategradient(problem,[], options); 
elseif strcmp(solver, 'SD')
   [T, ~, info] = steepestdescent(problem,[], options);
elseif strcmp(solver, 'RTR')
   [T, ~, info] = trustregions(problem, [], options);  
else
    error('Unvalid Solver!')
end

%% If we are changing a problem, the following is what to be changed
%% Cost function
% Here we use the entropy definition in M Cuturi (2013), NIPS
function [val, store] = cost(X,store)
    if ~all(isfield(store,{'logP', 'logX'}))
        logP = log(P);
        logX = log(X);
        store.logP = logP; 
        store.logX = logX;
    end  
    logX = store.logX;
    logP = store.logP;
    TrXC = sum(X .* (C-(lambda1*D)), 'all'); % the trace 
    %HX =  sum(X.*(log(X)./P),'all');   
    HX =  sum(X.*(logX - logP),'all'); 
    val = TrXC + HX*lambda2; % here it should be addition
end

%% Euclidient Gradient
function [egradval, store] = egrad(X, store)
    if ~all(isfield(store,{'logP', 'logX'}))
        logP = log(P);
        logX = log(X);
        store.logP = logP; 
        store.logX = logX;
    end     
    logX = store.logX;
    logP = store.logP;
    egradval = (C-lambda1*D)+lambda2*(ones(size(X))+logX-logP); % change to ones(size (X) if necessary 
end

%% Euclidient Hessian
function [ehessval] = ehess(X, eta)
   ehessval = lambda2*(eta./X);
end

end
