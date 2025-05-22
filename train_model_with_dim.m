function train_model_with_dim(num)
% num: 输入维度（滑动窗口大小）

% Load data
data = xlsread('data.xls');
data = data(1:60:end,1); % Take first column with 60-step sampling

%% Prepare input-output pairs using sliding window
n = length(data)-num;
x = [];
for i = 1:n
    x(:,i) = data(i:i+num); 
end
input = x(1:end-1,:);
output = x(end,:);

%% Train-test split
L = floor(size(input,2)*0.7);  
train_x = input(:,1:L);        
train_y = output(:,1:L);       
test_x = input(:,L+1:end);     
test_y = output(:,L+1:end);    

%% Normalization
[inputn_train,inputps] = mapminmax(train_x); 
inputn_test = mapminmax('apply',test_x,inputps);
[outputn_train,outputps] = mapminmax(train_y);
outputn_test = mapminmax('apply',test_y,outputps);

%% Reshape
M = size(train_x, 2); N = size(test_x, 2);
inputn_train = double(reshape(inputn_train, num, 1, 1, M));
inputn_test = double(reshape(inputn_test, num, 1, 1, N));
outputn_train = outputn_train'; outputn_test = outputn_test';

% Convert to sequence
Inputn_train = cell(M,1); Inputn_test = cell(N,1);
for i = 1:M, Inputn_train{i} = inputn_train(:, :, 1, i); end
for i = 1:N, Inputn_test{i} = inputn_test(:, :, 1, i); end

%% Network architecture
numFeatures = num;
numResponses = 1;
layers = [
    sequenceInputLayer(numFeatures)
    gruLayer(128)
    lstmLayer(64)
    dropoutLayer(0.5)
    lstmLayer(32,'OutputMode','last')
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    tanhLayer
    regressionLayer];

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
%% Train
fprintf('Training model with input dimension = %d ...\\n', num);
net = trainNetwork(Inputn_train, outputn_train, layers, options);

%% Save model
save(sprintf('trained_model_%d.mat', num), 'net', 'inputps', 'outputps');
fprintf('✅ Model saved as trained_model_%d.mat\\n', num);

end
