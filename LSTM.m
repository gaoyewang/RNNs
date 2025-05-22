clear; clc;

% === 1. 加载数据 ===
data = xlsread('data.xls');
data = data(1:60:end, 1);  % 取第一列，60步采样

%% 准备输入输出对（使用滑动窗口）
num = 8;   % 时间窗口大小（可以修改）
n = length(data) - num;
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

% === 3. 格式转换 ===
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);  % 转换为 cell 数组
    vt_train{i, 1} = t_train(:, i);  % 转换为 cell 数组
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);    % 转换为 cell 数组
    vt_test{i, 1} = t_test(:, i);    % 转换为 cell 数组
end

% === 4. 创建 LSTM 网络架构 ===
layers = [ ...
    sequenceInputLayer(8)              % 输入层
    lstmLayer(35)                      % LSTM 层
    reluLayer                           % ReLU 层
    fullyConnectedLayer(1)              % 回归层
    regressionLayer];

% === 5. 训练选项 ===
options = trainingOptions('adam', ...   % 梯度下降
    'MaxEpochs',300, ...                % 最大迭代次数
    'GradientThreshold',1, ...         % 梯度阈值 
    'InitialLearnRate',0.015,...
    'Verbose',0, ...
    'Plots','training-progress');            % 学习率

% === 6. 训练网络 ===
tic
net = trainNetwork(vp_train, vt_train, layers, options);
toc

% === 7. 预测 ===
t_sim1 = predict(net, vp_train);  % 训练集预测
t_sim2 = predict(net, vp_test);   % 测试集预测

% === 8. 数据反归一化 ===
T_sim1 = mapminmax('reverse', t_sim1, ps_output);  % 反归一化
T_sim2 = mapminmax('reverse', t_sim2, ps_output);  % 反归一化
T_train1 = train_y;
T_test2 = test_y;

T_sim2 = cell2mat(T_sim2);  % 转为矩阵
LSTM_TSIM2 = T_sim2';

% === 9. 可视化 ===
figure
plot(T_test2)
hold on
plot(T_sim2')
% === 9.1 计算回归评价指标 ===
y_true = T_test2(:);     % ground truth
y_pred = T_sim2(:);      % predicted

% 1. MSE
mse_val = mean((y_true - y_pred).^2);

% 2. RMSE
rmse_val = sqrt(mse_val);

% 3. MAE
mae_val = mean(abs(y_true - y_pred));

% 4. R² (Coefficient of Determination)
sst = sum((y_true - mean(y_true)).^2);   % 总平方和
sse = sum((y_true - y_pred).^2);          % 误差平方和
r2_val = 1 - (sse / sst);                 % 计算 R²

% 5. RPD (Residual Predictive Deviation)
std_ref = std(y_true);                               % 测试集真实值的标准差
rpd_val = std_ref / rmse_val;                        % RPD = std / RMSE

% === 打印指标结果 ===
fprintf('\n=== LSTM 模型测试集指标 ===\n');
fprintf('MSE   = %.4f\n', mse_val);
fprintf('RMSE  = %.4f\n', rmse_val);
fprintf('MAE   = %.4f\n', mae_val);
fprintf('R²    = %.4f\n', r2_val);   % 打印 R²
fprintf('RPD   = %.4f\n', rpd_val);   % 打印 RPD

% === 10. 保存模型 ===
net_lstm = net;  % 改名为 net_lstm
save('net_lstm.mat', 'net_lstm', 'ps_input', 'ps_output');  % 保存归一化的参数
