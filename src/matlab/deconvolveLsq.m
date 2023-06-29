function proportions = deconvolveLsq(cfDNA, atlasMeans, C)
%deconcolve Estimate tissue proportions using support vector regression
%   INPUTS:
%       cfDNA:     (vector N) of methylation values at N markers (mixture of T tissues)
%       atlasMeans:(matrix T x N) with the mean methylation value of each marker at each tissue
%       C:         (float) parameter of SVR, controls the effect of the regularization vs the mean squared error 
    x = atlasMeans';

    [Ngenes, Ntissues] = size(x);

    % set up quadratic program to solve support vector regression
    %P in python qp, quadratic component
    H = x' * x + eye(Ntissues) / C;
    
    
    % q in python qp, linear component
    f = -cfDNA * x;
    
    % no inequality constraints
    A = [];
    b = [];
    
    % equality constraints, proportions should add to 1
    Aeq = ones(1, Ntissues);
    beq = 1;
    
    % inequality constraints on the range of variable ([0,1])
    lb = zeros(1, Ntissues);
    ub = ones(1, Ntissues);
 
    
    % solve quadratic program, X contains the solution, fval other
    % information about convergence 
    [X,FVAL] = quadprog(H,f,A,b, Aeq, beq, lb, ub);

    % return the proportions - CHECK: X and proportions should be the same
    % here as there are no slack variables
    proportions = X(1:Ntissues);
end