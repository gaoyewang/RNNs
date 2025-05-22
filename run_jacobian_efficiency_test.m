function run_jacobian_efficiency_test(dim_list)
% 评估不同输入维度下，计算 Jacobian 和 C(k) 所需时间
% 要求每个维度已训练好模型，并保存在 trained_model_<num>.mat 中

eps = 1e-5;
time_log = [];

for idx = 1:length(dim_list)
    n = dim_list(idx);
    fprintf('\n--- 正在测试维度 %d 的收缩性计算时间 ---\n', n);
    
    % 初始化输入
    x = rand(n,1);
    
    % 加载模型
    load(sprintf('trained_model_%d.mat', n), 'net');
    
    % 初始化矩阵
    J = zeros(n,n);
    C = zeros(1,n);
    C_next = zeros(1,n);
    
    % 原始输出与下一状态
    y0 = predict(net, {x});
    x_next = [x(2:end); y0];
    y1 = predict(net, {x_next});

    % 计时
    tic;
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
    elapsed_time = toc;

    % 显示范数结果
    A = C_next * J * pinv(C);
    norm_A = norm(A, 2);
    norm_C = norm(C, 2);
    fprintf('‖A‖=%.4f, ‖C(k)‖=%.4f, 耗时=%.4f 秒\n', norm_A, norm_C, elapsed_time);

    % 记录
    time_log = [time_log; n, elapsed_time];
end

% 画图
figure;
scatter(time_log(:,1), time_log(:,2), 60, [0.2 0.4 0.8], 'filled');
xlabel('Input Dimension n');  % 修改为英文
ylabel('Computation Time (seconds)');  % 修改为英文
%title('Computation Time Comparison for Shrinkage Indicators Across Different Input Dimensions');  % 修改为英文
disp('Testing Complete.');
box on;
end
