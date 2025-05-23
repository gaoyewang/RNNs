
clear, close all
clc
data = xlsread('data.xls');
data = data(1:60:end, 1); 

% 1. 缺失值检查（强制处理）
if all(isfinite(data))
    disp('数据校验通过：未发现缺失值');
else
    missing_pos = find(~isfinite(data));
    error('数据存在%d个无效值（首例位置：%d），请预处理',...
          length(missing_pos), missing_pos(1));
end

% 2. 异常值检查（可配置处理方式）
Q = quantile(data, [0.25, 0.75]);
IQR = Q(2) - Q(1);
outlier_flags = (data < (Q(1)-1.5*IQR)) | (data > (Q(2)+1.5*IQR));
outlier_count = sum(outlier_flags);

% 异常值处理策略选择
if outlier_count > 0
    % ===== 策略选择 =====
    processing_mode = 'warning'; % 可选：'error'/'warning'/'auto_correct'
    % ===================
    
    outlier_pos = find(outlier_flags);
    msg = sprintf('发现%d个异常值（位置示例：%s）',...
                 outlier_count, mat2str(outlier_pos(1:min(3,end))));
    
    switch processing_mode
        case 'error'
            error('IQR准则异常值：%s', msg);
        case 'warning'
            warning('异常值已保留：%s', msg);
            % 保留原始数据
        case 'auto_correct'
            % 自动缩尾处理
            lb = Q(1)-1.5*IQR;
            ub = Q(2)+1.5*IQR;
            data(data < lb) = lb;
            data(data > ub) = ub;
            fprintf('已自动修正%d个异常值（缩尾处理）\n', outlier_count);
    end
else
    disp('数据校验通过：未发现异常值');
end

% 3. 增强版质量报告
fprintf('\n==== 数据质量报告 ====\n');
fprintf('样本量:    %d\n', length(data));
fprintf('缺失值:    %d个\n', sum(~isfinite(data)));
fprintf('异常值:    %d个（%.2f%%）\n',...
        outlier_count, outlier_count/length(data)*100);
if outlier_count > 0
    fprintf('处理方式:  %s\n', processing_mode);
end
fprintf('数据范围:  [%.2f, %.2f]\n', min(data), max(data));
fprintf('=====================\n');

% 后续处理保持不变...