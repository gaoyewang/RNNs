% Clear workspace
clear, close all
clc
warning off

% Load data
data = xlsread('data.xls');
data = data(1:60:end,1); % Take first column with 60-step sampling

%% Prepare input-output pairs using sliding window
num = 8;   % Time window size (can be modified)
n = length(data)-num;
for i = 1:n
    x(:,i) = data(i:i+num); 
end
input = x(1:end-1,:);
output = x(end,:);

%% Train-test split (70% training, 30% testing)
L = floor(size(input,2)*0.7);  
train_x = input(:,1:L);        
train_y = output(:,1:L);       
test_x = input(:,L+1:end);     
test_y = output(:,L+1:end);    

%% Data normalization
[inputn_train,inputps] = mapminmax(train_x); % Normalize to [-1,1]
inputn_test = mapminmax('apply',test_x,inputps);

[outputn_train,outputps] = mapminmax(train_y);
outputn_test = mapminmax('apply',test_y,outputps);

%% Reshape data for RNN
M = size(train_x, 2);
N = size(test_x, 2);

inputn_train = double(reshape(inputn_train, 8, 1, 1,M));
inputn_test = double(reshape(inputn_test, 8, 1, 1,N));

outputn_train = outputn_train';
outputn_test = outputn_test';

%% Convert to cell arrays for sequence data
for i = 1:M
    Inputn_train{i, 1} = inputn_train(:, :, 1, i);
end

for i = 1:N
    Inputn_test{i, 1} = inputn_test(:, :, 1, i);
end

%% Network architecture
numFeatures = 8; % = sliding window size
numResponses = 1;

layers = [...
    sequenceInputLayer(numFeatures,'Name','input')
    gruLayer(128,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
    lstmLayer(64,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
    dropoutLayer(0.5,'Name','drop2')
    lstmLayer(32,'OutputMode','last','Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
    dropoutLayer(0.5,'Name','drop3')
    fullyConnectedLayer(numResponses,'Name','fc')
    tanhLayer('Name','tanh_output')    
    regressionLayer('Name','output')];

%% Training options
if gpuDeviceCount>0
    mydevice = 'gpu';
else
    mydevice = 'cpu';
end

options = trainingOptions('adam', ...
    'L2Regularization',0.0001,...
    'MaxEpochs',200, ...
    'GradientThreshold',0.4, ...
    'InitialLearnRate',0.0005, ...
    'LearnRateSchedule',"piecewise", ...
    'LearnRateDropPeriod',100, ...
    'LearnRateDropFactor',0.3, ...
    'MiniBatchSize',16,...
    'Verbose',false, ...
    'Shuffle',"every-epoch",...
    'ExecutionEnvironment',mydevice,...
    'Plots','training-progress');

%% Train network
disp('Training IRNN network...')
net = trainNetwork(Inputn_train,outputn_train,layers, options);

%% Prediction
an0 = predict(net,Inputn_test); % Test set prediction
an1 = predict(net,Inputn_train); % Training set prediction

% Denormalize predictions
T_sim2 = mapminmax('reverse',an0,outputps); 

%% Visualization
analyzeNetwork(net)

figure
plot(test_y,'b-','LineWidth',1.5)
hold on
plot(T_sim2','r--','LineWidth',1.5)
%grid on

% Add proper English labels and legend
xlabel('Step k','FontSize',12)
ylabel('Value','FontSize',12)
%title('True Values vs Predicted Values','FontSize',14)
legend({'True Values','Predicted Values'},'FontSize',12,'Location','best')
% === Evaluate Performance on Test Set ===
y_true = test_y(:);       % Ground truth
y_pred = T_sim2(:);       % Predicted result

% 1. MSE
mse_val = mean((y_true - y_pred).^2);

% 2. RMSE
rmse_val = sqrt(mse_val);

% 3. MAE
mae_val = mean(abs(y_true - y_pred));

% 4. R² (Coefficient of Determination)
sst = sum((y_true - mean(y_true)).^2);   % Total sum of squares
sse = sum((y_true - y_pred).^2);          % Sum of squared errors
r2_val = 1 - (sse / sst);                 % Calculate R²

% 5. RPD (Residual Predictive Deviation)
std_ref = std(y_true);                               % Standard deviation of true values
rpd_val = std_ref / rmse_val;                        % RPD = std / RMSE

% === Print Results ===
fprintf('\n=== IRNN Hybrid Network Test Set Metrics ===\n');
fprintf('MSE   = %.4f\n', mse_val);
fprintf('RMSE  = %.4f\n', rmse_val);
fprintf('MAE   = %.4f\n', mae_val);
fprintf('R²    = %.4f\n', r2_val);   % Print R²
fprintf('RPD   = %.4f\n', rpd_val);   % Print RPD

net_irnn = net;  % 改名为 net_irnn
save('net_irnn.mat', 'net_irnn', 'inputps', 'outputps');
