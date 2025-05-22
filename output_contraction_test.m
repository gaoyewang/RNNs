clear; clc;

% 加载模型
load('trained_model.mat', 'net', 'inputps', 'outputps');

% 初始化输入
x = rand(8,1);
eps = 1e-5;
n = length(x);

% 初始化雅可比矩阵
J = zeros(n,n);
C = zeros(1,n);
C_next = zeros(1,n);

% 原始输出与下一状态
y0 = predict(net, {x});
x_next = [x(2:end); y0];
y1 = predict(net, {x_next});

% 数值估算雅可比
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

% 验证收缩条件
A = C_next * J * pinv(C);
norm_A = norm(A, 2);
beta = 0.01;

fprintf('‖C(k+1)*J(k)*pinv(C(k))‖ = %.6f\n', norm_A);
if norm_A <= sqrt(1 - beta)
    disp('✅ 满足输出收缩条件');
else
    disp('❌ 不满足输出收缩条件');
end

% 验证 C(k) 有界性
norm_C = norm(C, 2);
a = 2;

fprintf('‖C(k)‖ = %.6f\n', norm_C);
if norm_C <= a
    fprintf('✅ 满足 ‖C(k)‖ ≤ a = %.2f\n', a);
else
    fprintf('❌ 不满足 ‖C(k)‖ ≤ %.2f\n', a);
end

%% === 图像展示部分 ===
figure;

% First plot: Sensitivity of C(k) and C(k+1)
subplot(1,2,1);
plot(C, '-o', 'Color', [0, 0.3, 0.6], 'MarkerFaceColor', [0, 0.3, 0.6]); hold on;
plot(C_next, '-o', 'Color', [0.4, 0.6, 1], 'MarkerFaceColor', [0.4, 0.6, 1]);
xlabel('Step k', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Sensitivity', 'FontSize', 14, 'FontName', 'Times New Roman');
legend('C(k)', 'C(k+1)', 'Location', 'northeast');
% Move the title to the bottom
set(gca, 'Title', text('String', '(a) Sensitivity of C(k) and C(k+1)', 'Position', [0.5, -0.15, 0], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14, 'FontName', 'Times New Roman'));

% Second plot: J(k)
subplot(1,2,2);
bar(vecnorm(J), 'FaceColor', [0 0.4 1], 'BarWidth', 0.7);  % Column norms in a beautiful blue
xlabel('Step k', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('‖J(:,i)‖', 'FontSize', 14, 'FontName', 'Times New Roman');
% Move the title to the bottom
set(gca, 'Title', text('String', '(b) J(k)', 'Position', [0.5, -0.15, 0], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14, 'FontName', 'Times New Roman'));
box on;

% 1输出映射 C(k)、C(k+1) 条形图
% 第一个图：C(k) 和 C(k+1) 的灵敏度
figure;
plot(C, '-o', 'Color', [0, 0.3, 0.6], 'MarkerFaceColor', [0, 0.3, 0.6]); hold on;
plot(C_next, '-o', 'Color', [0.4, 0.6, 1], 'MarkerFaceColor', [0.4, 0.6, 1]);
xlabel('Step k', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Sensitivity', 'FontSize', 14, 'FontName', 'Times New Roman');
legend('C(k)', 'C(k+1)', 'Location', 'northeast');

% 第二个图：状态转移雅可比 J 的列范数
figure;
bar(vecnorm(J), 'FaceColor', [0 0.4 1], 'BarWidth', 0.7);  % Column norms in a beautiful blue
xlabel('Step k', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('‖J(:,i)‖', 'FontSize', 14, 'FontName', 'Times New Roman');
box on;

% 第三个图：关键矩阵范数对比
figure;
bar([norm(J), norm(C), norm(C_next), norm(A) ]);
set(gca, 'xticklabel', {'‖J(k)‖','‖C(k)‖','‖C(k+1)‖','‖A(k)‖'}, 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Value', 'FontSize', 14, 'FontName', 'Times New Roman');
box on;

figure;
subplot(1,2,1);
plot(C, '-o', 'Color', [0, 0.3, 0.6], 'MarkerFaceColor', [0, 0.3, 0.6]); hold on;
plot(C_next, '-o', 'Color', [0.4, 0.6, 1], 'MarkerFaceColor', [0.4, 0.6, 1]);
xlabel('Step k', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('Sensitivity', 'FontSize', 12, 'FontName', 'Times New Roman');
legend('C(k)', 'C(k+1)', 'Location', 'northeast');

% 2状态转移雅可比 J 的列范数（表示输入扰动的放大程度）
subplot(1,2,2);
bar(vecnorm(J), 'FaceColor', [0 0.4 1], 'BarWidth', 0.7);  % Column norms in a beautiful blue
xlabel('Step k', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('‖J(:,i)‖', 'FontSize', 12, 'FontName', 'Times New Roman');
box on;

% 3️ 关键矩阵范数对比
figure;
bar([norm(J), norm(C), norm(C_next), norm(A) ]);
set(gca,'xticklabel', {'‖J(k)‖','‖C(k)‖','‖C(k+1)‖','‖A(k)‖'});
ylabel('2-范数');
title('关键矩阵范数对比');
box on;

% % 4 映射方向图（示意 δx 被 C 映射压缩为 δy）
% % 只画前2维为例（1维输出也能看方向）
% dx_sample = randn(n,1); dx_sample = dx_sample / norm(dx_sample);
% dy_now = C * dx_sample;
% dy_next = C_next * dx_sample;
% 
% figure;
% plot([0 dx_sample(1)], [0 dx_sample(2)], 'b', 'LineWidth', 2); hold on;
% plot([0 dy_now], [0 0], 'r--', 'LineWidth', 2);
% plot([0 dy_next], [0 0], 'g--', 'LineWidth', 2);
% legend('扰动方向 δx', 'C(k)·δx', 'C(k+1)·δx');
% title('扰动映射方向示意（简化2维）');
% axis equal; grid on;
