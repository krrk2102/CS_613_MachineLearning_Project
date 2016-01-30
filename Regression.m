function y = Regression(Alpha, Flag, B, X, Y, Para1, x)
%--------------------------------------------------------------------------
% Regression.m
% Work with the function in file SVMNR.m
% Functionalities
% Input data a vector of x for regression 
% Generate expected output data y, a number
%--------------------------------------------------------------------------
% Input Parameters
% Alpha is vector of (alpha - alpha*) factor
% Flag is the vector of 1*l, with values from 0, 1, 2
% B constant factor of regression function
% X is original input data , n*l matrix, l for number of data samples
% Y is original expected output, 1*l vector, l for number of data samples
% Para1 is parameter for gaussian kernel functions. 
% x is data waiting to be applied by regression function, vector of 1*l
% y is output value

%%
%----------------Initializing Kernel Function Parameter--------------------
sigma = Para1;
%%
%%
%----------------------Input Data Normalization----------------------------
[X, minX, maxX] = premnmx(X);
x = 2 * ((x-minX)./(maxX-minX)) - 1;
[~, minY, maxY] = premnmx(Y);
%%
%%
%---------------------Calculate Output Values------------------------------
l = length(Alpha);
SUM = 0;
for i = 1 : l
    if Flag(i) > 0
        SUM = SUM + Alpha(i) * exp(-(norm(x-X(:, i)))^2/(2*sigma^2));
    end
end
y = SUM + B;
%%
%%
%--------------------Recover Data------------------------------------------
y = postmnmx(y, minY, maxY);