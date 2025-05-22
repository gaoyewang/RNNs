clear; clc;

% === 1. Load Trained Networks ===
load('net_irnn.mat', 'net_irnn');
load('net_lstm.mat', 'net_lstm');
load('net_trans.mat', 'net_transformer'); % Ensure net_trans is correctly loaded


% === 2. Construct Perturbed/Noisy Input ===
x_base = rand(1, 8);
perturb = 0.01 * randn(1, 8);
x_pert = x_base + perturb;
noise = 0.2 * randn(1, 8);
x_noise = x_base + noise;

% === 3. General Prediction Function (Fixed Syntax) ===
function y = predictAny(net, x)
    if isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork')
        inputLayer = net.Layers(1);
        expectedFeatureDim = inputLayer.InputSize;

        % Automatically adjust input shape
        if expectedFeatureDim == 1 && size(x,1) == 8
            x = x';
        elseif expectedFeatureDim == 8 && size(x,2) == 8
            x = x';
        end

        x_seq = {x};
        y = predict(net, x_seq);

        % Automatically retrieve value from cell
        if iscell(y)
            y = y{1};
        end

    elseif isa(net, 'dlnetwork')
        y = extractdata(predict(net, dlarray(single(x'), 'CB')));  % Handling for dlnetwork type
    else
        error('Unsupported network type.');
    end
end

% === 4. Model Perturbation Response Error ===
y_lstm_base = predictAny(net_lstm, x_base);
y_lstm_pert = predictAny(net_lstm, x_pert);
y_lstm_noise = predictAny(net_lstm, x_noise);
diff_lstm_pert = sum((y_lstm_pert - y_lstm_base).^2);
diff_lstm_noise = sum((y_lstm_noise - y_lstm_base).^2);

y_trans_base = predictAny(net_transformer, x_base);
y_trans_pert = predictAny(net_transformer, x_pert);
y_trans_noise = predictAny(net_transformer, x_noise);
diff_trans_pert = sum((y_trans_pert - y_trans_base).^2);
diff_trans_noise = sum((y_trans_noise - y_trans_base).^2);

y_irnn_base = predictAny(net_irnn, x_base);
y_irnn_pert = predictAny(net_irnn, x_pert);
y_irnn_noise = predictAny(net_irnn, x_noise);
diff_irnn_pert = sum((y_irnn_pert - y_irnn_base).^2);
diff_irnn_noise = sum((y_irnn_noise - y_irnn_base).^2);

% === 5. Bar Chart to Show the Effect of Perturbation and Noise ===
% figure(1);
% subplot(1,2,1);
% bar([diff_lstm_pert, diff_trans_pert, diff_irnn_pert]);
% set(gca, 'xticklabel', {'LSTM', 'Transformer', 'RNN'}); % Changed IRNN to RNN here
% ylabel('Perturbation Response Squared Difference'); title('Output Differences Under Perturbation');
% 
% subplot(1,2,2);
% bar([diff_lstm_noise, diff_trans_noise, diff_irnn_noise]);
% set(gca, 'xticklabel', {'LSTM', 'Transformer', 'RNN'}); % Changed IRNN to RNN here
% ylabel('Noise Response Squared Difference'); title('Output Differences Under Noise');

% === 6. Output Shrinkage Trend Plot (20 Perturbations) ===
figure(2);
steps = 1:30;
diffs_lstm = zeros(1,30);
diffs_trans = zeros(1,30);
diffs_irnn = zeros(1,30);

for i = 1:30
    x_now = x_base + 0.01 * randn(1, 8);  % Simulate input perturbation
    diffs_lstm(i) = sum((predictAny(net_lstm, x_now) - y_lstm_base).^2);  % Calculate error
    diffs_trans(i) = sum((predictAny(net_transformer, x_now) - y_trans_base).^2);
    diffs_irnn(i) = sum((predictAny(net_irnn, x_now) - y_irnn_base).^2);
end

% Plot the graph without specifying colors
plot(steps, diffs_lstm, '-o', 'LineWidth', 1.5); hold on;
plot(steps, diffs_trans, '-s', 'LineWidth', 1.5);
plot(steps, diffs_irnn, '-d', 'LineWidth', 1.5);

% Set legend and labels
xlabel('Step k');
ylabel('Output Error (L2)');
legend('LSTM', 'Transformer', 'RNN'); % Changed IRNN to RNN here
%title('Output Shrinkage Trend During Training (Simulation)');

% === 7. Comparison of Different Perturbation Intensities ===
% === 7. Comparison of Different Multiplicative Perturbation Intensities ===
figure(3);
steps = 30;  % Set number of steps
pert_levels = linspace(0.01, 0.1, steps);  % Increase perturbation intensity gradually
errs_lstm = zeros(size(pert_levels));
errs_trans = zeros(size(pert_levels));
errs_irnn = zeros(size(pert_levels));

for i = 1:steps
    % === 原先是加性扰动：dx = pert_levels(i) * randn(1, 8); ===
    ratio = pert_levels(i) * randn(1, 8);          % 比例扰动因子
    x_scaled = x_base .* (1 + ratio);              % 乘性缩放后的输入

    % 计算每个模型的输出误差
    errs_lstm(i)  = sum((predictAny(net_lstm, x_scaled)  - y_lstm_base ).^2);
    errs_trans(i) = sum((predictAny(net_transformer, x_scaled) - y_trans_base).^2);
    errs_irnn(i)  = sum((predictAny(net_irnn, x_scaled)  - y_irnn_base ).^2);
end

% === 绘图：L2误差随扰动强度的变化 ===
semilogx(pert_levels, errs_lstm, '-o', 'LineWidth', 1.5, 'DisplayName', 'LSTM');
hold on;
semilogx(pert_levels, errs_trans, '-s', 'LineWidth', 1.5, 'DisplayName', 'Transformer');
semilogx(pert_levels, errs_irnn, '-d', 'LineWidth', 1.5, 'DisplayName', 'RNN');

% X轴刻度与格式
xticks([pert_levels(1), pert_levels(5), pert_levels(10), ...
        pert_levels(15), pert_levels(20), pert_levels(25), pert_levels(end)]);
xtickformat('%.2f');

% 标注与图例
legend('Location', 'northwest');
xlabel('Multiplicative Perturbation Intensity (σ)');
ylabel('Output Error (L2)');
title('Model Response under Input Scaling Perturbation');

%title('Output Error Comparison at Different Perturbation Intensities');

initial_noise_level = 0.01;  % Initial noise level
max_noise_level = 1;         % Maximum noise level
num_steps = 30;              % Number of steps
noise_levels = linspace(initial_noise_level, max_noise_level, num_steps);  % Linearly increasing noise levels

% === 4. Store Errors ===
errs_lstm = zeros(1, num_steps);  % LSTM model errors
errs_trans = zeros(1, num_steps); % Transformer model errors
errs_irnn = zeros(1, num_steps);  % RNN model errors

% === 5. Calculate Errors at Different Noise Levels ===
for i = 1:num_steps
    noise = noise_levels(i) * randn(1, 8);  % Generate Gaussian noise
    x_noisy = x_base + noise;  % Add noise

    % Calculate LSTM model error
    errs_lstm(i) = sum((predictAny(net_lstm, x_noisy) - predictAny(net_lstm, x_base)).^2);

    % Calculate Transformer model error
    errs_trans(i) = sum((predictAny(net_transformer, x_noisy) - predictAny(net_transformer, x_base)).^2);

    % Calculate RNN model error
    errs_irnn(i) = sum((predictAny(net_irnn, x_noisy) - predictAny(net_irnn, x_base)).^2);
end

% === 6. Plot Error Graph for Different Noise Levels ===
figure;
plot(noise_levels, errs_lstm, '-o', 'LineWidth', 1.5, 'DisplayName', 'LSTM');
hold on;
plot(noise_levels, errs_trans, '-s', 'LineWidth', 1.5, 'DisplayName', 'Transformer');
plot(noise_levels, errs_irnn, '-d', 'LineWidth', 1.5, 'DisplayName', 'RNN'); % Changed IRNN to RNN here
xlabel('Noise Intensity (Standard Deviation)');
ylabel('Output Error (L2)');
%title('Output Error Comparison at Different Noise Intensities');
legend('show');
