function [T, info] = CouplingMatrix_symmLaplacian(C, Ls,Lt,SourceX, TargetX, n, m, p, q, lambda, alpha, eta, solver, options)
% This function is to define and solve the Laplacian regularized OT problem by using the optimization
% on the coupling matrix manifolds.
% The OT objective function is the version (20) defined in 
%
%   Nicolas Courty, Remi Flamary, Devis Tuia, and Alain Rakotomamonjy. Optimal transport for domain adaptation.
%      IEEE transactions on pattern analysis and machine intelligence 39(9):1853?1865, 2016
% 
% The usage
%    [T, info] = CouplingMatrix_Laplacian(C, n, m, p, q, lambda, alpha, eta, L, solver, opts)
%
% Inputs:
% C:  The positive cost matrix of size nxm
% n and m:  number of sources and number of targets for transport
% p:  The positive source allocation/distribution of size n
% q:  The positive target allocation of size m
%     if p or q are missing, its default values are 1/n*ones(n,1) and
%     1/m*ones(m,1)
% lambda: the regulasizer for the entropy regularizer, default = 0.1
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
    
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0.1;
end

if ~exist('eta', 'var') || isempty(eta)
    eta = 0.1;
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
    if ~all(isfield(store,{'logX', 'TTt', 'SSt'}))   % Remember one more variable
        store.logX = log(X);
        store.TTt = TargetX * TargetX';
        store.SSt = SourceX * SourceX';
        %a = 0; b =1; alpha = a+(b-a).*rand(1,1);      Why this???
    end  
    logX = store.logX;
    TTt = store.TTt;
    SSt = store.SSt;
    TrXC = sum(X .* C, 'all'); % the trace  
    HX =  sum(X.*logX,'all');  % - Junbin added
    Lap = sum( Ls .* (X *TTt * X'), 'all');
    lap_2 = (1-alpha)*Lap+ alpha* sum(Lt.*(X'*SSt*X),'all');
    val = TrXC + lambda * HX + 0.5*eta *lap_2; % here it should be addition
end

%% Euclidient Gradient
function [egradval, store] = egrad(X, store)
    if ~all(isfield(store,{'logX', 'TTt', 'SSt'})) 
        store.logX = log(X);
        store.TTt = TargetX * TargetX';
        store.SSt = SourceX * SourceX';
    end       
    logX = store.logX; 
    TTt = store.TTt;
    SSt = store.SSt;
    %egradval = C + lambda*log(X) + eta *(1-alpha)*Ls*X*TTt+ alpha*STt*X*Lt;   
    egradval = C + lambda*(ones(size(X))+logX) + eta * ((1-alpha)*Ls*X*TTt+ alpha*SSt*X*Lt);
end

%% Euclidient Hessian
function [ehessval] = ehess(X, xi)     
    TTt = TargetX * TargetX';
    SSt = SourceX * SourceX';
    ehessval = lambda*(xi./X) + eta*((1-alpha)*Ls*xi*TTt+alpha*SSt*xi*Lt);  % missing a bracket
    
end

end
