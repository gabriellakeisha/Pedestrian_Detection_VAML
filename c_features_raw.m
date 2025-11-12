% c_features_raw.m — extract full images features
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

posDir = 'images/pos';
negDir = 'images/neg';
targetSize = [128 64];

[X_raw, y, fileNames] = loadPedestrianDataset(posDir, negDir, 'raw', targetSize);
save('features/raw/features_raw.mat','X_raw','y','fileNames','targetSize','-v7.3');

fprintf('Raw features saved: %d samples × %d dimensions\n',size(X_raw,1),size(X_raw,2));
