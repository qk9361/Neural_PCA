
function [Wlda] = PCA(X, dims)
[X_row, ~] = size(X);

% step1. �������Ļ���ÿһά�����ݼ�ȥ��ά�ľ�ֵ
X = X - repmat(sum(X, 1) / X_row, [X_row, 1]);

% step2. �����һ������ÿ��������Ȩ�ض�һ��
d = sum(X.^2, 1) / X_row;
X = X ./ repmat((d.^0.5), [X_row, 1]);

% step3. ��Э�������C
C = X' * X / (X_row-1);

% step4. ����C����������������ֵ
[V1, D1] = eig(C);
[V, D] = cdf2rdf(V1, D1); %���ԽǾ���ת��Ϊʵ�ԽǾ���

% step5. �õ���������ֵ��Ӧ����������
[temp, IX] = max(D);
[~, ix] = sort(temp, 'descend');
len = min(dims, length(ix(1,:)));
Wlda = V(:, IX(ix(1: len)));
end