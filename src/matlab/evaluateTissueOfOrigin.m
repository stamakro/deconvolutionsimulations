function [pearsonr,rmse,mae,maxDev] = evaluateTissueOfOrigin(true, predicted)
%evaluateTissueOfOrigin calculate different evaluation metrics between true
%and predicted profiles 
%   Detailed explanation goes here

CC = corrcoef(true, predicted);

pearsonr = CC(1,2);

errors = true - predicted;

rmse = sqrt(mean(errors.^2));
mae = median(abs(errors));

maxDev = max(abs(errors));

end