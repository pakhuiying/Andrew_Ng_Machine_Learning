function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_list = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list = C_list;
error_list = zeros(length(C_list),length(sigma_list)); %fill in horizontally. sigma in inner loop
%put in matrix so u can use matrix indices to identify which C & sigma are
%optimal

for c = 1:length(C_list)
    for sig = 1:length(sigma_list)
        C = C_list(c);
        sigma = sigma_list(sig);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval);
        error_list(c,sig) = mean(double(predictions ~= yval)); %compute error
    end
end

%M = min(error_list,[],'all')  %finds the minimum over all elements 
%[M_sig, I_sig] = min(error_list,[], 2);
%[row, col] = find(error_list==M)
%C = C_list(row) %use index of matrix of 
%sigma = sigma_list(col)
[M, I] = min(error_list(:)) %change error list to one long vector. this swaps the C & sigma index
%it arranges column wise into vectors instead of row wise

C_index = ceil(I/length(sigma_list)) %round up cus index starts at 1 instead of 0
sigma_index = mod(I,length(sigma_list))

if sigma_index == 0
    sigma_index = length(sigma_list); %if mod =0, means it's at the last index
end

C = C_list(sigma_index)
sigma = sigma_list(C_index)
% =========================================================================

end
