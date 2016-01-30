function [Alpha1, Alpha2, Alpha, Flag, B] = SVMNR(X, Y, Epsilon, C, Para1)
%%
% SVMNR.m
% Support Vector Machine for Nonlinear Regression
%%
% X Input data matrix n*l, l for number of data samples
% Y Ideal output data vector, 1*l, l for number of data samples
% Epsilon 
% C is constant factor
% Kernel function is gaussian
% Para1 is sigma in kernel
% Generate result with function in Regression.m file
% ??????
% Alpha1 alpha factor
% Alpha2 alpha* factor
% Alpha (alpha - alpha*) factor
% Flag 1*l vector, 0 for non support vector, 1 for margin support vector, 2
% for standard support vector
% B constant in regression function
%--------------------------------------------------------------------------

%%
%-----------------------Data normalizatio----------------------------------
nntwarn off
X = premnmx(X);
Y = premnmx(Y);
%%
%%
%-----------------------Set Parameter for Kernel Function------------------
sigma = Para1;
%%
%%
%-----------------------Construct K Matrix---------------------------------
l = size(X, 2);
K = zeros(l, l);
for i = 1 : l
    for j = 1 : l
        x = X(:, i);
        y = X(:, j);
        K(i, j) = exp(-(norm(x-y))^2/(2*sigma^2));
    end
end
%%
%%
%------------Quadprog Parameters H,Ft,Aeq,Beq,lb,ub------------------------
% Factor of regression functions should be solved by quadprog
H = [K, -K; -K, K];
H = (H + H') / 2;
Ft = [Epsilon * ones(1, l) - Y, Epsilon * ones(1, l) + Y];
Aeq = [ones(1, l), -ones(1, l)];
Beq = 0;
lb = eps .* ones(2*l, 1);
ub = C * ones(2*l, 1);
%%
%%
%--------------Solve Quadprog Problem--------------------------------------
OPT = optimset;
OPT.LargeScale = 'off';
OPT.Display = 'off';
[Gamma,~] = quadprog(H, Ft, [], [], Aeq, Beq, lb, ub, [], OPT);
%%
%%
%------------------------Calculate Factors in Regression Function----------
Alpha1 = (Gamma(1:l, 1))';
Alpha2 = (Gamma((l+1):end, 1))';
Alpha = Alpha1 - Alpha2;
Flag = 2 * ones(1, l);
%%
%%
%-------------------Classification on Support Vectors----------------------
Err = 0.000000000001;
for i = 1 : l
    AA = Alpha1(i);
    BB = Alpha2(i);
    if (abs(AA - 0) <= Err) && (abs(BB - 0) <= Err)
        Flag(i) = 0;
    end
    if (AA > Err) && (AA < C - Err) && (abs(BB - 0) <= Err)
        Flag(i) = 2;
    end
    if (abs(AA - 0) <= Err) && (BB > Err) && (BB < C - Err)
        Flag(i) = 2;
    end
    if (abs(AA - C) <= Err) && (abs(BB - 0) <= Err)
        Flag(i) = 1;
    end
    if (abs(AA - 0) <= Err) && (abs(BB - C) <= Err)
        Flag(i) = 1;
    end
end
%%
%%
%--------------------Calculate Constant Factor B---------------------------
B = 0;
counter = 0;
for i = 1 : l
    AA = Alpha1(i);
    BB = Alpha2(i);
    if (AA > Err) && (AA < C - Err) && (abs(BB - 0) <= Err)
        % Calculate weights for support vectors.
        SUM = 0;
        for j = 1 : l
            if Flag(j) > 0
                SUM = SUM + Alpha(j) * exp(-(norm(X(:, j)-X(:, i)))^2/(2*sigma^2));
            end
        end
        b = Y(i) - SUM - Epsilon;
        B = B + b;
        counter = counter + 1;
    end
    if (abs(AA - 0) <= Err) && (BB > Err) && (BB < C - ERR)
        SUM = 0;
        for j = 1 : l
            if Flag(j) > 0
                SUM = SUM + Alpha(j) * exp(-(norm(X(:, j)-X(:, i)))^2/(2*sigma^2));
            end
        end
        b = Y(i) - SUM + Epsilon;
        B = B + b;
        counter = counter + 1;
    end
end
if counter == 0
    B = 0;
else
    B = B / counter;
end

