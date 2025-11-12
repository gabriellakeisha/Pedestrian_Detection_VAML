%% cross_validation.m - Proper K-Fold Cross-Validation
% Compares models using k-fold CV with within-fold hyperparameter tuning
clearvars; close all; clc; rng(1234,'twister'); 
addpath(genpath('src'));

fprintf('K-FOLD CROSS-VALIDATION MODEL COMPARISON\n');

% Load features
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

X_raw = double(R.X_raw);
X_hog = double(H.X_hog);
y = double(R.y);

% Configuration
k_folds = 5;  % Standard choice - fixed for all experiments
N = numel(y);

fprintf('Dataset: %d samples, %d features\n', N, size(X_hog, 2));
fprintf('Using %d-fold cross-validation\n\n', k_folds);

% Create fold indices
indices = crossvalind('Kfold', N, k_folds);

% Define models to compare
models_to_test = {
    'KNN-HOG', 'hog', 'knn';
    'SVM-Linear-HOG', 'hog', 'svm_linear';
    'SVM-RBF-HOG', 'hog', 'svm_rbf'
};

results = struct();

%% Run Cross-Validation for Each Model
for m = 1:size(models_to_test, 1)
    model_name = models_to_test{m, 1};
    feature_type = models_to_test{m, 2};
    classifier_type = models_to_test{m, 3};
    
    fprintf('=== %s ===\n', model_name);
    
    % Select features
    if strcmp(feature_type, 'hog')
        X = X_hog;
    else
        X = X_raw;
    end
    
    % Storage for fold results
    fold_acc = zeros(k_folds, 1);
    fold_prec = zeros(k_folds, 1);
    fold_rec = zeros(k_folds, 1);
    fold_f1 = zeros(k_folds, 1);
    fold_time = zeros(k_folds, 1);
    
    % Run each fold
    for fold = 1:k_folds
        fprintf('  Fold %d/%d... ', fold, k_folds);
        fold_start = tic;
        
        % Split data for this fold
        test_idx = (indices == fold);
        train_idx = ~test_idx;
        
        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_test = X(test_idx, :);
        y_test = y(test_idx);
        
        % Normalize within fold (important!)
        [X_train_norm, X_test_norm] = normalize_features(X_train, X_test);
        
        try
            % Train and predict with within-fold hyperparameter tuning
            switch classifier_type
                case 'knn'
                    [mdl, best_params] = train_knn(X_train_norm, y_train);
                    
                case 'svm_linear'
                    [mdl, best_params] = train_svm_linear(X_train_norm, y_train);
                    
                case 'svm_rbf'
                    [mdl, best_params] = train_svm_rbf(X_train_norm, y_train);
            end
            
            % Predict
            y_pred = predict(mdl, X_test_norm);
            
            % Calculate metrics
            [acc, prec, rec, f1] = calculate_metrics(y_test, y_pred);
            
            fold_acc(fold) = acc;
            fold_prec(fold) = prec;
            fold_rec(fold) = rec;
            fold_f1(fold) = f1;
            fold_time(fold) = toc(fold_start);
            
            fprintf('Acc: %.2f%% (K=%d)\n', acc, best_params.K);
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            fold_acc(fold) = NaN;
            fold_prec(fold) = NaN;
            fold_rec(fold) = NaN;
            fold_f1(fold) = NaN;
            fold_time(fold) = toc(fold_start);
        end
    end
    
    % Store results for this model
    results(m) = compile_results(model_name, fold_acc, fold_prec, fold_rec, fold_f1, fold_time);
    fprintf('  → Mean Accuracy: %.2f%% (±%.2f%%)\n\n', ...
        results(m).mean_acc, results(m).std_acc);
end

%% Display Comprehensive Results
display_detailed_results(results, k_folds);

%% Visualization
plot_cv_results(results, k_folds);

%% Save Results
save_cv_results(results);

fprintf('Cross-validation complete! Best model: %s\n', get_best_model(results));

%% Helper Functions

function [X_train_norm, X_test_norm] = normalize_features(X_train, X_test)
    % Normalize features: z-score normalization
    mu = mean(X_train);
    sigma = std(X_train);
    sigma(sigma < eps) = 1;  % Avoid division by zero
    
    X_train_norm = (X_train - mu) ./ sigma;
    X_test_norm = (X_test - mu) ./ sigma;
end

function [mdl, best_params] = train_knn(X_train, y_train)
    % KNN with within-fold hyperparameter tuning
    k_candidates = [1, 3, 5, 7, 9, 11];
    best_k = 5;
    best_score = 0;
    
    % Simple validation split for quick tuning
    cv = cvpartition(y_train, 'Holdout', 0.2);
    X_tr = X_train(cv.training, :);
    y_tr = y_train(cv.training);
    X_val = X_train(cv.test, :);
    y_val = y_train(cv.test);
    
    for k = k_candidates
        temp_mdl = fitcknn(X_tr, y_tr, 'NumNeighbors', k, ...
                          'Distance', 'euclidean', 'Standardize', false);
        y_pred = predict(temp_mdl, X_val);
        score = sum(y_pred == y_val) / numel(y_val);
        
        if score > best_score
            best_score = score;
            best_k = k;
        end
    end
    
    % Train final model on full training data
    mdl = fitcknn(X_train, y_train, 'NumNeighbors', best_k, ...
                 'Distance', 'euclidean', 'Standardize', false);
    best_params.K = best_k;
end

function [mdl, best_params] = train_svm_linear(X_train, y_train)
    % Linear SVM with within-fold hyperparameter tuning
    c_candidates = [0.1, 1, 10, 100];
    best_c = 1;
    best_score = 0;
    
    cv = cvpartition(y_train, 'Holdout', 0.2);
    X_tr = X_train(cv.training, :);
    y_tr = y_train(cv.training);
    X_val = X_train(cv.test, :);
    y_val = y_train(cv.test);
    
    for c = c_candidates
        temp_mdl = fitcsvm(X_tr, y_tr, 'KernelFunction', 'linear', ...
                          'BoxConstraint', c, 'Standardize', false);
        y_pred = predict(temp_mdl, X_val);
        score = sum(y_pred == y_val) / numel(y_val);
        
        if score > best_score
            best_score = score;
            best_c = c;
        end
    end
    
    mdl = fitcsvm(X_train, y_train, 'KernelFunction', 'linear', ...
                 'BoxConstraint', best_c, 'Standardize', false);
    best_params.C = best_c;
    best_params.K = NaN; % For consistent output
end

function [mdl, best_params] = train_svm_rbf(X_train, y_train)
    % RBF SVM with within-fold hyperparameter tuning
    c_candidates = [0.1, 1, 10, 100];
    best_c = 1;
    best_score = 0;
    
    cv = cvpartition(y_train, 'Holdout', 0.2);
    X_tr = X_train(cv.training, :);
    y_tr = y_train(cv.training);
    X_val = X_train(cv.test, :);
    y_val = y_train(cv.test);
    
    for c = c_candidates
        temp_mdl = fitcsvm(X_tr, y_tr, 'KernelFunction', 'rbf', ...
                          'BoxConstraint', c, 'KernelScale', 'auto', ...
                          'Standardize', false);
        y_pred = predict(temp_mdl, X_val);
        score = sum(y_pred == y_val) / numel(y_val);
        
        if score > best_score
            best_score = score;
            best_c = c;
        end
    end
    
    mdl = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', ...
                 'BoxConstraint', best_c, 'KernelScale', 'auto', ...
                 'Standardize', false);
    best_params.C = best_c;
    best_params.K = NaN;
end

function [acc, prec, rec, f1] = calculate_metrics(y_true, y_pred)
    % Calculate classification metrics
    tp = sum((y_pred == 1) & (y_true == 1));
    fp = sum((y_pred == 1) & (y_true == 0));
    tn = sum((y_pred == 0) & (y_true == 0));
    fn = sum((y_pred == 0) & (y_true == 1));
    
    acc = (tp + tn) / (tp + fp + tn + fn) * 100;
    prec = tp / (tp + fp + eps);
    rec = tp / (tp + fn + eps);
    f1 = 2 * (prec * rec) / (prec + rec + eps);
end

function result = compile_results(name, acc, prec, rec, f1, time)
    % Compile results for a model
    result.name = name;
    result.mean_acc = mean(acc, 'omitnan');
    result.std_acc = std(acc, 'omitnan');
    result.mean_prec = mean(prec, 'omitnan');
    result.std_prec = std(prec, 'omitnan');
    result.mean_rec = mean(rec, 'omitnan');
    result.std_rec = std(rec, 'omitnan');
    result.mean_f1 = mean(f1, 'omitnan');
    result.std_f1 = std(f1, 'omitnan');
    result.mean_time = mean(time, 'omitnan');
    result.fold_acc = acc;
    result.fold_prec = prec;
    result.fold_rec = rec;
    result.fold_f1 = f1;
end

function display_detailed_results(results, k_folds)
    % Display comprehensive results table
    fprintf('\n%s\n', repmat('=', 100, 1));
    fprintf('CROSS-VALIDATION RESULTS SUMMARY (%d-FOLD)\n', k_folds);
    fprintf('%s\n', repmat('=', 100, 1));
    
    fprintf('%-20s | %8s | %12s | %12s | %12s | %8s\n', ...
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time(s)');
    fprintf('%s\n', repmat('-', 100, 1));
    
    for m = 1:numel(results)
        fprintf('%-20s | %5.2f±%.2f | %5.3f±%.3f | %5.3f±%.3f | %5.3f±%.3f | %7.2f\n', ...
            results(m).name, ...
            results(m).mean_acc, results(m).std_acc, ...
            results(m).mean_prec, results(m).std_prec, ...
            results(m).mean_rec, results(m).std_rec, ...
            results(m).mean_f1, results(m).std_f1, ...
            results(m).mean_time);
    end
    fprintf('%s\n\n', repmat('-', 100, 1));
end

function plot_cv_results(results, k_folds)
    % Create visualization plots
    fig = figure('Position', [100, 100, 1200, 500], 'Name', 'Cross-Validation Results');
    
    % Plot 1: Accuracy distribution across folds
    subplot(1, 2, 1);
    acc_data = [];
    for m = 1:numel(results)
        acc_data = [acc_data, results(m).fold_acc];
    end
    
    boxplot(acc_data, 'Labels', {results.name});
    ylabel('Accuracy (%)');
    title(sprintf('%d-Fold CV: Accuracy Distribution', k_folds));
    grid on;
    xtickangle(45);
    
    % Plot 2: Mean performance comparison
    subplot(1, 2, 2);
    metrics = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
    mean_values = [
        [results.mean_acc] / 100;  % Convert to 0-1 scale
        [results.mean_prec];
        [results.mean_rec];
        [results.mean_f1]
    ];
    
    b = bar(mean_values');
    set(gca, 'XTickLabel', {results.name});
    legend(metrics, 'Location', 'best');
    ylabel('Score');
    title('Mean Performance Metrics');
    grid on;
    ylim([0, 1]);
    xtickangle(45);
    
    % Save figure
    if ~exist('results/figures', 'dir')
        mkdir('results/figures');
    end
    saveas(fig, 'results/figures/cross_validation_results.png');
end

function save_cv_results(results)
    % Save results to file
    if ~exist('results', 'dir')
        mkdir('results');
    end
    
    save('results/cross_validation_results.mat', 'results');
    fprintf('Results saved to results/cross_validation_results.mat\n');
end

function best_model_name = get_best_model(results)
    % Find the best performing model based on mean accuracy
    [~, best_idx] = max([results.mean_acc]);
    best_model_name = results(best_idx).name;
end