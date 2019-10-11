
function [Wlda] = PCA(X, dims)
[X_row, ~] = size(X);

% step1. 特征中心化，每一维的数据减去该维的均值
X = X - repmat(sum(X, 1) / X_row, [X_row, 1]);

% step2. 方差归一化，让每个特征的权重都一样
d = sum(X.^2, 1) / X_row;
X = X ./ repmat((d.^0.5), [X_row, 1]);

% step3. 求协方差矩阵C
C = X' * X / (X_row-1);

% step4. 计算C的特征向量和特征值
[V1, D1] = eig(C);
[V, D] = cdf2rdf(V1, D1); %复对角矩阵转化为实对角矩阵

% step5. 得到最大的特征值对应的特征向量
[temp, IX] = max(D);
[~, ix] = sort(temp, 'descend');
len = min(dims, length(ix(1,:)));
Wlda = V(:, IX(ix(1: len)));
end