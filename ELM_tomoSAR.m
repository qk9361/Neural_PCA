% Author: Qian Kun
% Kernel ELM for TomoSAR inversion test
clear all
close all
clc

addpath('/Users/qiankun/Desktop/Master Arbeit/ELM_MA/test_chaolake')
load kpca_nn_simulation.mat

% simulated SAR signals
g = G1';

% simulated steering vectors
for i = 1:1000
    r(i,:) = reshape(R(:,i,:), 26, 1);
end

% generate training and testing data
g_train = g(1:900,:);
r_train = r(1:900,:);

g_test = g(901:end,:);
r_test = r(901:end,:);

r_pred = Gauss(g_test, g_train) * conj((Gauss(g_train, g_train))') * r_train;
r_pred1 = r_pred(:,1:13)';
r_test1 = r_test(:,1:13)';

r_pred2 = r_pred(:,13:end)';
r_test2 = r_test(:,13:end)';

angbias1 = acosd( abs(sum(r_pred1.*conj(r_test1)))./ sum(abs(r_pred1).*abs(r_test1 ) ) );
angbias2 = acosd( abs(sum(r_pred2.*conj(r_test2)))./ sum(abs(r_pred2).*abs(r_test2 ) ) );
%angbias = angbias / size(r_test, 1);