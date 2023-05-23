
function proportions = deconvolve(cfDNA, atlasMeans, C, epsilon)
%deconcolve Estimate tissue proportions using support vector regression
%   INPUTS:
%       cfDNA:     (vector N) of methylation values at N markers (mixture of T tissues)
%       atlasMeans:(matrix N x T) with the mean methylation value of each marker at each tissue
%       C:         (float) parameter of SVR, controls the effect of the regularization vs the constraints 
%       epsilon:   (float) parameter of SVR, if prediction error is less than epsilon for a gene, then that gene does not contribute to the margin  

    x = atlasMeans';

    [Ngenes, Ntissues] = size(x);

    Nvariables = Ngenes + Ntissues; % (w_i and ksis)

    % set up quadratic program to solve support vector regression

    %P in python qp, quadratic component
    H = zeros(Nvariables, Nvariables);
    H(1:Ntissues, 1:Ntissues) = eye(Ntissues);
    
    % q in python qp, linear component
    f = ones(1,Nvariables) * C;
    f(1:Ntissues) = 0;
    
    % G = A, h = b, inequality constraints 
    A1 = [-x -eye(Ngenes)];
    b1 = epsilon - cfDNA;
    
    A2 = [x -eye(Ngenes)];
    b2 = epsilon + cfDNA;
     
    
    A = [A1; A2];
    b = [b1 b2];
    
    % equality constraint
    Aeq = [ones(1, Ntissues) zeros(1,Ngenes)];
    beq = 1;
    
    % inequality constraints on the range of the values: slack variables
    % non-negative, proportions between 0 and 1
    infinity = 1e5;
    lb = zeros(1, Nvariables);
    ub = ones(1, Nvariables) * infinity;
    ub(1:Ntissues) = 1;
    
    % solve quadratic program, X contains the solution, fval other
    % information about convergence
    [X,FVAL] = quadprog(H,f,A,b, Aeq, beq, lb, ub);

    % ignore slack variables and only keep the tissue proportions
    proportions = X(1:Ntissues);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

