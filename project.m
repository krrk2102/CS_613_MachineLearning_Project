clear;
rawData = csvread('data_msft.csv',1,1);
rawData = rawData(:, 6:-1:5);
X = zeros(8, size(rawData, 1)-25);
Y = zeros(2, size(rawData, 1)-25);
k = 1;
for i = size(rawData, 1)-20 : -1 : 6
    EMA15 = rawData(i+15, 1);
    factor = 2 / (15+1);
    for j = 15 : -1 : 1
        EMA15 = EMA15 + factor * (rawData(i+j, 1) - EMA15);
    end
    X(1, k) = rawData(i, 1) - EMA15;
    X(2, k) = (rawData(i, 1) - rawData(i+5, 1)) / rawData(i+5, 1);
    X(3, k) = (rawData(i, 2) - rawData(i+5, 2)) / rawData(i+5, 2);
    X(4, k) = (rawData(i, 1) - rawData(i+10, 1)) / rawData(i+10, 1);
    X(5, k) = (rawData(i, 2) - rawData(i+10, 2)) / rawData(i+10, 2);
    X(6, k) = (rawData(i, 1) - rawData(i+15, 1)) / rawData(i+15, 1);
    X(7, k) = (rawData(i, 1) - rawData(i+20, 1)) / rawData(i+20, 1);
    X(8, k) = i;
    EMA3 = rawData(i+3, 1);
    EMA35 = rawData(i-5+3, 1);
    factor = 2 / (3+1);
    for j = 3 : -1 : 1
        EMA3 = EMA3 + factor * (rawData(i+j, 1) - EMA3);
        EMA35 = EMA35 + factor * (rawData(i-5+j, 1) - EMA35);
    end
    Y(1, k) = (EMA35 - EMA3) / EMA3; 
    Y(2, k) = i;
    k = k + 1;
end

tp = zeros(5, 1);
tn = zeros(5, 1);
fp = zeros(5, 1);
fn = zeros(5, 1);
accuracy = zeros(5, 1);
profit_std = zeros(5, 1);
price_std = zeros(5, 1);
fold = 1;

mse = zeros(5,1);
nmse = zeros(5,1);
mae = zeros(5,1);

for fold = 1 : 5
    trainX = X;
    trainY = Y;
    testX = X(:, floor(size(X,2)*0.2)*(fold-1)+1:floor(size(X,2)*0.2)*fold);
    testY = Y(:, floor(size(Y,2)*0.2)*(fold-1)+1:floor(size(Y,2)*0.2)*fold);
    trainX(:, floor(size(X,2)*0.2)*(fold-1)+1:floor(size(X,2)*0.2)*fold) = [];
    trainY(:, floor(size(Y,2)*0.2)*(fold-1)+1:floor(size(Y,2)*0.2)*fold) = [];
    C = 10;
    epsilon = 0.01;
    Sigma = 0.8;
    [Alpha1, Alpha2, Alpha, Flag, B] = SVMNR(trainX(1:end-1,:), trainY(1,:), epsilon, C, Sigma);
    pred_profit = zeros(2, size(testX, 2)-5);
    for i = 6 : size(testX, 2)
        pred_profit(1, i-5) = Regression(Alpha, Flag, B, trainX(1:end-1,:), trainY(1,:), Sigma, testX(1:end-1, i));
        pred_profit(2, i-5) = testX(size(testX,1), i);
    end
    pred_prices = zeros(size(testX, 2)-5, 1);
    for i = 6 : size(testX, 2)
        pred_prices(i-5, 1) = rawData(testX(size(testX,1),i), 1) * (1 + pred_profit(1, i-5));
    end
    figure;
    plot(1:size(testY, 2)-5,pred_profit(1,:),'r-',1:size(testY, 2)-5,testY(1,6:end),'b-.');
    legend('predicted profit', 'target profit');
    title(['Cross validation on profit for fold ', num2str(fold)]);
    figure;
    plot(1:size(testX, 2)-5,pred_prices,'r-',1:size(testX, 2)-5,rawData(testX(size(testX,1), 1:end-5)-5, 1),'b-.');
    legend('predicted price', 'target price');
    title(['Cross validation on price for fold ', num2str(fold)]);
    for i = 1 : size(pred_profit, 2)
        if pred_profit(1,i) > 0 
            if testY(1,i) > 0
                tp(fold,1) = tp(fold,1) + 1;
            else
                fp(fold,1) = fp(fold,1) + 1;
            end
        else
            if testY(1,i) > 0
                fn(fold,1) = fn(fold,1) + 1;
            else
                tn(fold,1) = tn(fold,1) + 1;
            end
        end
        mse(fold,1) = mse(fold,1)+(testY(1,i)-pred_profit(1,i)).^2;
        mae(fold,1) = mae(fold,1)+abs((testY(1,i)-pred_profit(1,i)));
    end
    accuracy(fold,1) = (tp(fold,1) + tn(fold,1)) / (tp(fold,1) + tn(fold,1) + fp(fold,1) + fn(fold,1));
    profit_std(fold,1) = std(pred_profit(1,:) - testY(1,6:end));
    price_std(fold,1) = std(pred_prices - rawData(testX(size(testX,1), 1:end-5)-5, 1));
    mse(fold,1) = mse(fold,1)/size(pred_profit, 2);
    nmse(fold,1)= mse(fold,1)/var(transpose(testY(1,6:end)));
    mae(fold,1) = mae(fold,1)/size(pred_profit, 2);
    
end
display(accuracy);
display(mse);
display(nmse);
display(mae);