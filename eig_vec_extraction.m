clear all
close all
clc

features = h5read('sim_data.h5','/features');
labels = h5read('sim_data.h5', '/labels');

features = features';
labels = labels';

[eig_vec, score, latent] = pca(features);

% y_hat = features * eig_vec;
% 
dif = score(:,1:2) - labels;
% 
dif(1:10,:)

