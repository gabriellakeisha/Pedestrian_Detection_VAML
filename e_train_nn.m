% e_train_nn.m — store NN models 
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

S = load('splits/splits.mat');
trainIdx = S.trainIdx;

R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

Xtr_raw = double(R.X_raw(trainIdx,:));
ytr = double(R.y(trainIdx));
Xtr_hog = double(H.X_hog(trainIdx,:));

% normalisation
mu_raw = mean(Xtr_raw);
sigma_raw = std(Xtr_raw);
sigma_raw(sigma_raw < eps) = 1;

mu_hog = mean(Xtr_hog);
sigma_hog = std(Xtr_hog);
sigma_hog(sigma_hog < eps) = 1;

Xtr_raw = (Xtr_raw - mu_raw) ./ sigma_raw;
Xtr_hog = (Xtr_hog - mu_hog) ./ sigma_hog;

% store models
modelNN_raw.neighbours = Xtr_raw;
modelNN_raw.labels = ytr;
modelNN_raw.mu = mu_raw;
modelNN_raw.sigma = sigma_raw;

modelNN_hog.neighbours = Xtr_hog;
modelNN_hog.labels = ytr;
modelNN_hog.mu = mu_hog;
modelNN_hog.sigma = sigma_hog;

save('models/nn/modelNN_raw.mat', 'modelNN_raw');
save('models/nn/modelNN_hog.mat', 'modelNN_hog');

fprintf('NN normalized models saved:\n');
fprintf('Full images/RAW: %d×%d (normalized)\n', size(Xtr_raw,1), size(Xtr_raw,2));
fprintf('HOG: %d×%d (normalized)\n', size(Xtr_hog,1), size(Xtr_hog,2));
