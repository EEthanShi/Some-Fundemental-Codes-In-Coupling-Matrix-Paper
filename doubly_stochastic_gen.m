function [B, d_1, d_2] = doubly_stochastic_gen(C, p, q, maxiter)
% Project a matrix to the generalised doubly stochastic matrices (Modified Sinkhorn's algorithm)
%
% function B = doubly_stochastic_gen(C, p, q)
% function B = doubly_stochastic(C, p, q, maxiter) 
%
% Given an element-wise non-negative matrix C of size nxm, returns a
% generalised doubly-stochastic matrix B of size nxm such that row sum is p and 
% the column sum is q by applying modified Sinkhorn's algorithm
% to C such that B can be regarded as the projection of C onto the
% generalised doubly-stochastic matrix
% 
% maxiter (optional): strictly positive integer representing the maximum 
%	number of iterations of modified Sinkhorn's algorithm. 
%	The default value of maxiter is nxm.

% The file is based on the standard Sinkhorn's algorithm from manopt toolbox 
% the description is developed based on the research paper
% Philip A. Knight, "The Sinkhorn–Knopp Algorithm: Convergence and 
% Applications" in SIAM Journal on Matrix Analysis and Applications 30(1), 
% 261-275, 2008.
%
% Please cite the Manopt paper as well as the research paper.

% This file may be part of Manopt: www.manopt.org.
% Original author: David Young, September 10, 2015.
% Contributors: Ahmed Douik, March 15, 2018.
% Change log:

    n = size(C, 1);
    m = size(C, 2);
    tol = eps(n);
    
    if ~exist('maxiter', 'var') || isempty(maxiter)
        maxiter = n*m;
    end
    
    iter = 1;
    d_1 = q(:)' ./sum(C);
    d_2 = p(:) ./(C * d_1.');
    while iter < maxiter
         iter = iter + 1;
        row = d_2.' * C;
        if  max(abs(row .* d_1 - q(:)')) <= tol
            % Do we really need to make the other condition Bx1 = p?  -
            % Ethan commented
            break;
        end
        d_1 =q(:)' ./row;
        d_2 = p(:)./(C * d_1.');
    end
    
    B = C .* (d_2 * d_1);     
end
