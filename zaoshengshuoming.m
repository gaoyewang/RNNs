clear; clc; 

% === 加载模型 ===
load('trained_model.mat', 'net', 'inputps', 'outputps');

% === 基准输入 ===
x_base = rand(8,1);

% === 参数设置 ===
noise_levels = linspace(0, 1, 20);      % 噪声水平范围
variation_levels = linspace(0, 1, 20);     % 变异水平范围
eps = 1e-5;
n = length(x_base);

% === 辅助函数：计算矩阵 ===
function [J, C, C_next, A] = calculate_matrices(net, x, eps)
    n = length(x);
    J = zeros(n,n);
    C = zeros(1,n);
    C_next = zeros(1,n);

    % 原始输出与下一状态
    y0 = predict(net, {x});
    x_next = [x(2:end); y0];
    y1 = predict(net, {x_next});

    for i = 1:n
        dx = zeros(n,1); dx(i) = eps;
        x_pert = x + dx;
        y_pert = predict(net, {x_pert});
        next_pert = [x_pert(2:end); y_pert];

        J(:,i) = (next_pert - x_next) / eps;
        C(i) = (y_pert - y0) / eps;

        x_next_pert = x_next + J(:,i)*eps;
        y_next_pert = predict(net, {x_next_pert});
        C_next(i) = (y_next_pert - y1) / eps;
    end

    A = C_next * J * pinv(C);
end

% === 噪声敏感性分析 ===
norm_J_noise = zeros(size(noise_levels));
norm_C_noise = zeros(size(noise_levels));
norm_Cnext_noise = zeros(size(noise_levels));
norm_A_noise = zeros(size(noise_levels));

for idx = 1:length(noise_levels)    
    noise = noise_levels(idx) * randn(n,1);
    x = x_base + noise;
    [J, C, C_next, A] = calculate_matrices(net, x, eps);
    
    norm_J_noise(idx) = norm(J,2);
    norm_C_noise(idx) = norm(C,2);
    norm_Cnext_noise(idx) = norm(C_next,2);
    norm_A_noise(idx) = norm(A,2);
end

% === 噪声敏感性绘图 ===
figure;
plot(noise_levels, norm_J_noise, '-o', 'LineWidth', 1.5); hold on;
plot(noise_levels, norm_C_noise, '-s', 'LineWidth', 1.5);
plot(noise_levels, norm_Cnext_noise, '-^', 'LineWidth', 1.5);
plot(noise_levels, norm_A_noise, '-d', 'LineWidth', 1.5);

xlabel('Noise Standard Deviation');
ylabel('2-Norm Value');
legend('‖J(k)‖','‖C(k)‖','‖C(k+1)‖','‖A(k)‖','Location','northwest');
%title('Sensitivity of Contraction Metrics under Input Noise');

% === 数据变异敏感性分析 ===
norm_J_var = zeros(size(variation_levels));
norm_C_var = zeros(size(variation_levels));
norm_Cnext_var = zeros(size(variation_levels));
norm_A_var = zeros(size(variation_levels));

for idx = 1:length(variation_levels)
    x_var = x_base .* (1 + variation_levels(idx) * randn(n,1));
    [J, C, C_next, A] = calculate_matrices(net, x_var, eps);
    
    norm_J_var(idx) = norm(J,2);
    norm_C_var(idx) = norm(C,2);
    norm_Cnext_var(idx) = norm(C_next,2);
    norm_A_var(idx) = norm(A,2);
end

% === 数据变异绘图 ===
figure;
plot(variation_levels, norm_J_var, '-o', 'LineWidth', 1.5); hold on;
plot(variation_levels, norm_C_var, '-s', 'LineWidth', 1.5);
plot(variation_levels, norm_Cnext_var, '-^', 'LineWidth', 1.5);
plot(variation_levels, norm_A_var, '-d', 'LineWidth', 1.5);

xlabel('Variation Level');
ylabel('2-Norm Value');
legend('‖J(k)‖','‖C(k)‖','‖C(k+1)‖','‖A(k)‖','Location','northwest');
%title('Sensitivity of Contraction Metrics under Input Variation');
