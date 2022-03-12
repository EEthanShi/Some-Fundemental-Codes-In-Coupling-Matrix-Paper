function M = couplingmatrixfactory(n, m, p, q)
% Manifold of n-by-m coupling matrix manifold with positive entries.
%
% function M = couplingmatrixfactory(n, m, p, q)
%
% The returned structure M is a Manopt manifold structure to optimize over
% the set of n-by-m matrices with (strictly) positive entries and such that
% the column sum to p and the row sum to q, and sum(p) = sum(q).  
%
% The metric imposed on the manifold is the Fisher metric such that 
% the set of n-by-m coupling matrices is a Riemannian submanifold of 
% the space of n-by-m matrices.
% Also it should be noted that the retraction operation that we define 
% is first order and as such the checkhessian tool cannot verify 
% the slope correctly at non-critical points.
%             
% The file is based on developments in the research paper
% Junbin Gao, Ethan Shi, Xia Hong, XXX XXX and Daming Shi,
% "Coupling Matrix Manifold and Its Applications in Optimal Transport
% (tentative)" to be published on arXiv.
%
% Link to the paper: http://arxiv.org/abs/ .
%
% Please cite the Manopt paper as well as the research paper:
% @Article{sun2015multinomial,
%   author  = {Junbin Gao, Ethan Shi, Xia Hong, XXX XXX and Daming Shi},
%   title   = {Coupling Matrix Manifold and Its Applications in Optimal Transport},
%   journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
%   year    = {XXXX},
%   volume  = {42},
%   number  = { },
%   pages   = { },
%   doi     = {10.1109/TPAMI.xxxxx}
% }

% This file is part of Manopt: www.manopt.org.
% 
% Contributors:
% Change log:
%
%    Sep. 6, 2019 (NB):
%        Removed M.exp() as it was not implemented.

    if ~exist('p', 'var') || isempty(p)
        p = ones(n,1)/n;
    end
    
    if ~exist('q', 'var') || isempty(q)
        q = ones(m,1)/m;
    end
    
   % if abs(1 - sum(p)/ sum(q)) > 10*eps
   %     error('p and q are not coupled, sum(p) equal to sum(q), please!')
   % end
    
    M.name = @() sprintf('%dx%d coupling matrices with positive entries', n, m);
    
    M.dim = @() (n-1)*(m-1);
    
    % We impose the Fisher metric.
    M.inner = @iproduct;
    function ip = iproduct(X, eta, zeta)
        ip = sum((eta(:).*zeta(:))./(X(:)));   
    end
    
    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
    
    M.dist = @(X, Y) error('couplingmatrixfactory.dist not implemented yet.');
    
    %M.typicaldist = @() error('couplingmatrixfactory.typicaldist not implemented yet.');  
    % It seems this line causes an error. Is this due to ManOpt?   
    
    
    % Column vectors of ones of length n and m. 
    e1 = ones(n, 1);
    e2 = ones(m, 1);
    Q = diag(1./q);
    
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad)
        eta = X .* egrad;
        rgrad = M.proj(X, eta);
        %lambda = -sum(X.*egrad, 1); % Row vector of length m.
        %rgrad = X.*egrad + (e*lambda).*X; % This is in the tangent space.
    end
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, xi)
        %Q = diag(1./q);
        XQ = X*Q;
       
        % Riemannian gradient computation.
        mu = pinv(diag(p) - XQ * X');
        eta = X .* egrad;
        alpha = mu*(eta * e2 - XQ*eta'*e1); % Row vector of length m.
        beta = Q*eta' *e1 - XQ' * alpha;
        gamma = eta - (alpha * e2' + e1 * beta') .* X;  % Riemannian gradient 
         
        dotmu = mu*(XQ*xi' + xi * XQ')*mu;
        % I need know how to apply ehess on xi
        doteta = ehess .* X + egrad .* xi;  %%%%
        
        dotalpha = dotmu*(eta * e2 - XQ * (eta' * e1)) + mu *(doteta * e2 - xi * Q * (eta' * e1) - XQ *doteta' *e1);
        dotbeta = Q * doteta' * e1 - Q*xi'*alpha - XQ' * dotalpha;
        dotgamma = doteta - (dotalpha*e2' + e1*dotbeta').*X - (alpha * e2' + e1 * beta') .* xi;
        
        % Correction term because of the non-constant metric that we
        % impose. The computation of the correction term follows the use of
        % Koszul formula.
        correction_term = - 0.5*(eta.*gamma)./(X);    
        rhess = dotgamma + correction_term;
        
        % Finally, projection onto the tangent space.
        rhess = M.proj(X, rhess);
    end
    
    % Projection of the vector eta in the ambeint space onto the tangent
    % space.
    M.proj = @projection;
    function etaproj = projection(X, eta)
        %Q = diag(1./q);
        XQ = X*Q;
        B = pinv(diag(p) - XQ * X');
        
        alpha = B*(eta * e2 - XQ*eta'*e1); % Row vector of length m.
        beta = Q*eta' *e1 - XQ' * alpha;
        etaproj = eta - (alpha * e2' + e1 * beta') .* X;
    end
    
    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;
    
    M.retr = @retraction;
    function Y = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        % A first-order retraction.
        YY = X.*exp(t*(eta./(X))); % Based on mapping for positive scalars.
                                        
        %if sum(isnan(YY),'all')
        %    disp('NaN in Retraction')
        %end
        Y = doubly_stochastic_gen(YY, p, q);
        % Projection onto the constraint set by Sinkhorn algorithm.
        
        % For numerical reasons, so that we avoid entries going to zero:
        
        Y = max(Y, eps/1e+60);
    end
    
    
    M.hash = @(X) ['z' hashmd5(X(:))];
    
    M.rand = @random;
    function X = random()
        % A random point in the ambient space.
        X = rand(n, m); %
        
        X = doubly_stochastic_gen(X, p, q);
        % For numerical reasons, so that we avoid entries going to zero:
        X = max(X, eps/1e+60);
    end
    
    M.randvec = @randomvec;
    function eta = randomvec(X)
        % A random vector in the tangent space
        eta = randn(n, m);
        eta = M.proj(X, eta); % Projection onto the tangent space.
        nrm = M.norm(X, eta);
        eta = eta / nrm;
    end
    
    M.lincomb = @matrixlincomb;
    
    M.zerovec = @(X) zeros(n, m);
    
    M.transp = @(X1, X2, d) projection(X2, d);
    
    % vec and mat are not isometries, because of the scaled metric.
    M.vec = @(X, U) U(:);
    M.mat = @(X, u) reshape(u, n, m);
    M.vecmatareisometries = @() false;
end
