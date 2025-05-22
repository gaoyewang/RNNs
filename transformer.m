clear; clc;

% === 1. 加载数据 ===
data = xlsread('data.xls');
data = data(1:60:end, 1);  % 取第一列，60步采样

%% 使用前num个时刻 预测下一个时刻
num = 8;   % 可修改
n = length(data)-num;
for i = 1:n
    x(:,i) = data(i:i+num); 
end
input = x(1:end-1,:);  % 输入数据（去掉最后一行）
output = x(end,:);     % 输出数据（最后一行）

%% 训练集与测试集划分
L = floor(size(input,2)*0.7);  % 70%用于训练，30%用于测试
train_x = input(:,1:L);        % 训练输入样本
train_y = output(1:L);         % 训练输出样本
test_x = input(:,L+1:end);     % 测试输入样本
test_y = output(L+1:end);      % 测试输出样本
M = size(train_x, 2);
N = size(test_x, 2);

% === 2. 数据归一化 ===
[p_train, ps_input] = mapminmax(train_x, 0, 1);  % 归一化到 [0, 1]
p_test = mapminmax('apply', test_x, ps_input);

[t_train, ps_output] = mapminmax(train_y, 0, 1);  % 对输出进行归一化
t_test = mapminmax('apply', test_y, ps_output);

%%  数据平铺

for i = 1:size(p_train, 2)
    trainD{i, :} = (reshape(p_train(:, i), 8, []));  % 将输入数据按时序拆分为cell数组
end

for i = 1:size(p_test, 2)
    testD{i, :} = (reshape(p_test(:, i), 8, []));  % 测试集也要转换成cell数组
end

targetD = t_train';  % 训练目标值
targetD_test = t_test';  % 测试目标值

% === 3. 创建 Transformer 网络架构 ===
numChannels = 8;
maxPosition = 256;
numHeads = 4;
numKeyChannels = numHeads * 32;

layers = [ 
    sequenceInputLayer(numChannels, Name="input")
    positionEmbeddingLayer(numChannels, maxPosition, Name="pos-emb")
    additionLayer(2, Name="add")
    selfAttentionLayer(numHeads, numKeyChannels, 'AttentionMask', 'causal')
    selfAttentionLayer(numHeads, numKeyChannels)
    indexing1dLayer("last")
    fullyConnectedLayer(1)
    regressionLayer];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, "input", "add/in2");

% === 4. 训练选项 ===
maxEpochs = 50;
miniBatchSize = 32;
learningRate = 0.001;
solver = 'adam';
shuffle = 'every-epoch';
gradientThreshold = 10;
executionEnvironment = "auto"; % chooses local GPU if available, otherwise CPU

options = trainingOptions(solver, ...
    'Plots', 'none', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', shuffle, ...
    'InitialLearnRate', learningRate, ...
    'GradientThreshold', gradientThreshold, ...
    'ExecutionEnvironment', executionEnvironment);

% === 5. 网络训练 ===
tic
net0 = trainNetwork(trainD, targetD, lgraph, options);
toc

% === 6. 预测 ===
t_sim1 = predict(net0, trainD);  % 训练集预测
t_sim2 = predict(net0, testD);   % 测试集预测

% === 7. 数据反归一化 ===
T_sim1 = mapminmax('reverse', t_sim1, ps_output);  % 反归一化
T_sim2 = mapminmax('reverse', t_sim2, ps_output);  % 反归一化
T_train1 = train_y;
T_test2 = test_y;

T_sim2 = double(T_sim2);

% === 8. 确保 T_test2 和 T_sim2 是列向量 ===
T_test2 = T_test2(:);  % 转换为列向量
T_sim2 = T_sim2(:);    % 转换为列向量

% === 10. 可视化 ===
figure
plot(T_test2)
hold on
plot(T_sim2')
% === 9. 指标计算 ===
y_true = T_test2(:);   % ground truth
y_pred = T_sim2(:);    % predicted

% 1. MSE
mse_val = mean((y_true - y_pred).^2);

% 2. RMSE
rmse_val = sqrt(mse_val);

% 3. MAE
mae_val = mean(abs(y_true - y_pred));

% 4. RPD (Residual Predictive Deviation)
rmse_val = sqrt(mean((y_true - y_pred).^2));         % 确保你有 RMSE
std_ref = std(y_true);                               % 测试集真实值的标准差
rpd_val = std_ref / rmse_val;                        % RPD = std / RMSE

% R² (Coefficient of Determination)
sst = sum((y_true - mean(y_true)).^2);   % 总平方和
sse = sum((y_true - y_pred).^2);          % 误差平方和
r2_val = 1 - (sse / sst);                 % 计算 R²

% === 打印指标结果 ===
fprintf('\n=== Transformer 模型测试集指标 ===\n');
fprintf('MSE   = %.4f\n', mse_val);
fprintf('RMSE  = %.4f\n', rmse_val);
fprintf('MAE   = %.4f\n', mae_val);
fprintf('RPD   = %.4f\n', rpd_val);   % 打印 RPD
fprintf('R²    = %.4f\n', r2_val);   % 打印 R²


% === 11. 保存模型 ===
net_transformer = net0;  % 改名为 net_transformer
save('net_trans.mat', 'net_transformer', 'ps_input', 'ps_output');  % 保存归一化的参数
