function [dis,T] = CM_OP(X,Y,a,b,lambda1,lambda2,delta,options)
% Compute the Order-Preserving Wasserstein Distance (OPW) for two sequences
% X and Y

% -------------
% INPUT:
% -------------
% X: a N * d matrix, representing the input sequence consists of of N
% d-dimensional vectors, where N is the number of instances (vectors) in X,
% and d is the dimensionality of instances;
% Y: a M * d matrix, representing the input sequence consists of of N
% d-dimensional vectors, , where N is the number of instances (vectors) in
% Y, and d is the dimensionality of instances;
% iterations = total number of iterations
% a: a N * 1 weight vector for vectors in X, default uniform weights if input []
% b: a M * 1 weight vector for vectors in Y, default uniform weights if input []
% lamda1: the weight of the IDM regularization, default value: 50
% lamda2: the weight of the KL-divergence regularization, default value:
% 0.1
% delta: the parameter of the prior Gaussian distribution, default value: 1
% VERBOSE: whether display the iteration status, default value: 0 (not display)

% -------------
% OUTPUT
% -------------
% dis: the OPW distance between X and Y
% T: the learned transport between X and Y, which is a N*M matrix


% -------------
% c : barycenter according to weights
% ADVICE: divide M by median(M) to have a natural scale
% for lambda

% -------------
% Copyright (c) 2017 Bing Su, Gang Hua
% -------------
%
% -------------
% License
% The code can be used for research purposes only.

    if nargin<5 || isempty(lambda1)
        lambda1 = 50;
    end

    if nargin<6 || isempty(lambda2)
        lambda2 = 0.1;
    end

    if nargin<7 || isempty(delta)
        delta = 1;
    end

    if nargin<8 || isempty(options)
        options.checkperiod = 1;
        options.maxiter = 100;
        options.verbosity = 2;
    end

    %tolerance=.5e-2;
    %maxIter= 20;
    % The maximum number of iterations; with a default small value, the
    % tolerance and VERBOSE may not be used;
    % Set it to a large value (e.g, 1000 or 10000) to obtain a more precise
    % transport;
    %p_norm=inf;

    N = size(X,1);
    M = size(Y,1);
    dim = size(X,2);
    if size(Y,2)~=dim
        disp('The dimensions of instances in the input sequences must be the same!');
    end
    
    v1=[1:N]';  
    v2=[1:M];
    r1 = v1/N;
    r2 = v2/M;    
    W1 = repmat(r1, 1, M);
    W2 = repmat(r2, N, 1);
    W3=W1-W2;
    D = 1./((W3).^2+1); % this is D; 
    
    %mid_para = sqrt((1/(N^2) + 1/(M^2)));
    mid_para =  (1/(N^2) + 1/(M^2));
    P = exp( - W3.^2/ (2*delta^2 * mid_para))/(delta*sqrt(2*pi));

    C = pdist2(X,Y, 'sqeuclidean');
    %D = D/(10^2);
    % In cases the instances in sequences are not normalized and/or are very
    % high-dimensional, the matrix D can be normalized or scaled as follows:
    % D = D/max(max(D));  D = D/(10^2);

    if isempty(a)
        a = ones(N,1)./N;
    end

    if isempty(b)
        b = ones(M,1)./M;
    end


    solver = 'CG';
    [T, ~] = CouplingMatrix_Order_perserving(C, N, M, a, b, lambda1,lambda2, D,P,solver,options);
    %T(1,2:end) = 0.0;
    %T(2:end,1) = 0.0;
    dis = sum(T .* C, 'all');
end