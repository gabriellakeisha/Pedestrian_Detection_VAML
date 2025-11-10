% g_train_knn.m — train KNN 
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));
if ~exist('models/knn','dir'), mkdir('models/knn'); end

% load and split the data
S = load('splits/splits.mat'); trainIdx = S.trainIdx;
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

% build the train-only sets
Xtr_raw = double(R.X_raw(trainIdx,:)); 
ytr = double(R.y(trainIdx));
Xtr_hog = double(H.X_hog(trainIdx,:));

% internal validation split (80/20)
Ntr = numel(ytr); 
idx = randperm(Ntr);
Nval = round(0.2*Ntr); 
valSel = idx(1:Nval); 
subSel = idx(Nval+1:end);

Xsub_raw = Xtr_raw(subSel,:); ysub = ytr(subSel);
Xval_raw = Xtr_raw(valSel,:); yval = ytr(valSel);
Xsub_hog = Xtr_hog(subSel,:);
Xval_hog = Xtr_hog(valSel,:);

% normalisation
mu_raw = mean(Xsub_raw);
sigma_raw = std(Xsub_raw);
sigma_raw(sigma_raw < eps) = 1;

mu_hog = mean(Xsub_hog);
sigma_hog = std(Xsub_hog);
sigma_hog(sigma_hog < eps) = 1;

% normalise validation data
Xsub_raw_norm = (Xsub_raw - mu_raw) ./ sigma_raw;
Xval_raw_norm = (Xval_raw - mu_raw) ./ sigma_raw;
Xsub_hog_norm = (Xsub_hog - mu_hog) ./ sigma_hog;
Xval_hog_norm = (Xval_hog - mu_hog) ./ sigma_hog;

% grid search parameters
Kset = [1 3 5 7 9 15 21 31];
Dset = {'euclidean', 'cityblock', 'cosine'};

bestRaw.acc = -inf; 
bestHog.acc = -inf;

fprintf('\nKNN GRID SEARCH\n');

% fulll images
fprintf('\nRAW Features\n');
for d = 1:numel(Dset)
    for k = 1:numel(Kset)
        mdl = fitcknn(Xsub_raw_norm, ysub, ...
            'NumNeighbors', Kset(k), ...
            'Distance', Dset{d}, ...
            'Standardize', false); 
        
        pred = predict(mdl, Xval_raw_norm);
        acc = mean(pred==yval)*100;
        
        if acc > bestRaw.acc
            bestRaw.acc = acc; 
            bestRaw.K = Kset(k); 
            bestRaw.dist = Dset{d};
            fprintf('  New best: K=%d, dist=%s -> %.2f%%\n', Kset(k), Dset{d}, acc);
        end
    end
end

% HOG
fprintf('\nHOG Features\n');
for d = 1:numel(Dset)
    for k = 1:numel(Kset)
        mdl = fitcknn(Xsub_hog_norm, ysub, ...
            'NumNeighbors', Kset(k), ...
            'Distance', Dset{d}, ...
            'Standardize', false);
        
        pred = predict(mdl, Xval_hog_norm);
        acc = mean(pred==yval)*100;
        
        if acc > bestHog.acc
            bestHog.acc = acc; 
            bestHog.K = Kset(k); 
            bestHog.dist = Dset{d};
            fprintf('New best: K=%d, dist=%s -> %.2f%%\n', Kset(k), Dset{d}, acc);
        end
    end
end

fprintf('\nFINAL BEST PARAMETERS\n');
fprintf('Best RAW: K=%d, dist=%s, val=%.2f%%\n', bestRaw.K, bestRaw.dist, bestRaw.acc);
fprintf('Best HOG: K=%d, dist=%s, val=%.2f%%\n', bestHog.K, bestHog.dist, bestHog.acc);

% compute normalisation
mu_raw_full = mean(Xtr_raw);
sigma_raw_full = std(Xtr_raw);
sigma_raw_full(sigma_raw_full < eps) = 1;

mu_hog_full = mean(Xtr_hog);
sigma_hog_full = std(Xtr_hog);
sigma_hog_full(sigma_hog_full < eps) = 1;

% normalise full training sets
Xtr_raw_norm = (Xtr_raw - mu_raw_full) ./ sigma_raw_full;
Xtr_hog_norm = (Xtr_hog - mu_hog_full) ./ sigma_hog_full;

% train final models on full training set
fprintf('\nTraining final models on full training set\n');
modelKNN_raw = fitcknn(Xtr_raw_norm, ytr, ...
    'NumNeighbors', bestRaw.K, ...
    'Distance', bestRaw.dist, ...
    'Standardize', false);

modelKNN_hog = fitcknn(Xtr_hog_norm, ytr, ...
    'NumNeighbors', bestHog.K, ...
    'Distance', bestHog.dist, ...
    'Standardize', false);

% save
save('models/knn/modelKNN_raw.mat', 'modelKNN_raw', 'bestRaw', ...
     'mu_raw_full', 'sigma_raw_full', '-v7.3');
save('models/knn/modelKNN_hog.mat', 'modelKNN_hog', 'bestHog', ...
     'mu_hog_full', 'sigma_hog_full', '-v7.3');