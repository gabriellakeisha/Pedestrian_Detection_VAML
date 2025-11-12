%% n_all_evaluation.m — Complete testing with all validation methods
clearvars; close all; clc; rng(1234,'twister'); addpath(genpath('src'));

fprintf('ALL EVALUATION PEDESTRIAN DETECTION TESTING\n');

%% 1. Load data for different validation methods
fprintf('Loading datasets for different validation methods...\n');

% 70/30 split
S70 = load('splits/splits.mat');
R = load('features/raw/features_raw.mat');
H = load('features/hog/features_hog.mat');

% 50/50 split  
try
    S50 = load('splits/splits_50_50.mat');
    has_50 = true;
catch
    has_50 = false;
    fprintf('50/50 split not found, skipping\n');
end

fprintf('Dataset loaded: %d total samples\n', numel(R.y));

%% 2. Test all models with complete metrics
models = {
    'NN-RAW', 'NN-HOG', ...
    'KNN-RAW', 'KNN-HOG', ...
    'SVM-Linear-RAW', 'SVM-Linear-HOG', ...
    'SVM-RBF-RAW', 'SVM-RBF-HOG'
};

%% Test with 70/30 split
fprintf('TESTING WITH 70/30 HOLDOUT SPLIT\n');

results_70 = test_all_models(models, R, H, S70.trainIdx, S70.testIdx, '70/30 Holdout');

%% Test with 50/50 split if available
if has_50
    fprintf('\n%s\n', repmat('=', 60, 1));
    fprintf('TESTING WITH 50/50 HOLDOUT SPLIT\n');
    fprintf('%s\n', repmat('=', 60, 1));
    
    results_50 = test_all_models(models, R, H, S50.trainIdx, S50.testIdx, '50/50 Holdout');
else
    results_50 = [];
end

%% Test with 5-Fold Cross Validation
fprintf('TESTING WITH 5-FOLD CROSS VALIDATION\n');

results_cv = run_cross_validation(models, R, H);

%% 3. Display comprehensive results tables for all methods
display_all_results(results_70, results_50, results_cv, has_50);

%% 4. Create comprehensive visualizations for all methods
create_comprehensive_plots_all_methods(results_70, results_50, results_cv, has_50);

%% 5. Statistical analysis and comparison
perform_comprehensive_statistical_analysis(results_70, results_50, results_cv, has_50);

%% HELPER FUNCTIONS 

function results = test_all_models(models, R, H, trainIdx, testIdx, split_name)
    Xte_raw = double(R.X_raw(testIdx,:)); 
    yte = double(R.y(testIdx));
    Xte_hog = double(H.X_hog(testIdx,:));
    
    fprintf('Testing %d models on %s split (%d test samples)...\n', ...
        numel(models), split_name, numel(yte));
    
    results = struct();
    all_predictions = zeros(numel(yte), numel(models));
    
    for i = 1:numel(models)
        model_name = models{i};
        fprintf('\n %s \n', model_name);
        
        try
            % Extract feature type from model name
            if contains(model_name, 'HOG')
                Xte = Xte_hog;
                feature_type = 'hog';
            else
                Xte = Xte_raw;
                feature_type = 'raw';
            end
            
            % Load appropriate model and normalize
            [mdl, mu, sigma] = load_model(model_name, feature_type);
            Xte_norm = (Xte - mu) ./ sigma;
            
            % Predict
            tic;
            if contains(model_name, 'NN')
                % NN requires special prediction
                y_pred = predict_nn(Xte_norm, mdl, model_name);
            else
                y_pred = predict(mdl, Xte_norm);
            end
            time_per_sample = toc / numel(yte);
            
            % Store predictions for confusion matrices
            all_predictions(:, i) = y_pred;
            
            % Calculate comprehensive metrics
            [accuracy, precision, recall, f1, tp, fp, tn, fn] = ...
                calculate_comprehensive_metrics(yte, y_pred);
            
            % Store results
            results(i).name = model_name;
            results(i).accuracy = accuracy;
            results(i).precision = precision;
            results(i).recall = recall;
            results(i).f1 = f1;
            results(i).tp = tp;
            results(i).fp = fp;
            results(i).tn = tn;
            results(i).fn = fn;
            results(i).time_per_sample = time_per_sample;
            results(i).split = split_name;
            
            fprintf('Accuracy: %.2f%%, Precision: %.3f, Recall: %.3f, F1: %.3f\n', ...
                accuracy, precision, recall, f1);
            fprintf('TP: %d, FP: %d, TN: %d, FN: %d\n', tp, fp, tn, fn);
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            % Store NaN values for failed models
            results(i).name = model_name;
            results(i).accuracy = NaN;
            results(i).precision = NaN;
            results(i).recall = NaN;
            results(i).f1 = NaN;
            results(i).tp = NaN;
            results(i).fp = NaN;
            results(i).tn = NaN;
            results(i).fn = NaN;
            results(i).time_per_sample = NaN;
            results(i).split = split_name;
        end
    end
    
    % Create confusion matrices for this validation method
    create_detailed_confusion_matrices(results, yte, all_predictions, split_name);
end

function results_cv = run_cross_validation(models, R, H)
    % Load full dataset
    X_raw = double(R.X_raw);
    X_hog = double(H.X_hog);
    y = double(R.y);
    
    N = numel(y);
    k_folds = 5;
    
    fprintf('Running %d-fold cross-validation on %d samples...\n', k_folds, N);
    
    % Create fold indices
    indices = crossvalind('Kfold', N, k_folds);
    
    results_cv = struct();
    
    for m = 1:numel(models)
        model_name = models{m};
        fprintf('\n%s... ', model_name);
        
        if contains(model_name, 'HOG')
            X = X_hog;
            feature_type = 'hog';
        else
            X = X_raw;
            feature_type = 'raw';
        end
        
        % Storage for fold results
        fold_acc = zeros(k_folds, 1);
        fold_prec = zeros(k_folds, 1);
        fold_rec = zeros(k_folds, 1);
        fold_f1 = zeros(k_folds, 1);
        fold_time = zeros(k_folds, 1);
        
        % Run each fold
        for fold = 1:k_folds
            % Split data
            test_idx = (indices == fold);
            train_idx = ~test_idx;
            
            X_train = X(train_idx, :);
            y_train = y(train_idx);
            X_test = X(test_idx, :);
            y_test = y(test_idx);
            
            % Normalize
            mu = mean(X_train);
            sigma = std(X_train);
            sigma(sigma < eps) = 1;
            
            X_train_norm = (X_train - mu) ./ sigma;
            X_test_norm = (X_test - mu) ./ sigma;
            
            % Train and predict
            try
                tic;
                
                if contains(model_name, 'NN')
                    % For NN, we need to create model from training data
                    mdl_cv.neighbours = X_train_norm;
                    mdl_cv.labels = y_train;
                    y_pred = predict_nn(X_test_norm, mdl_cv, model_name);
                else
                    % Retrain the model with best parameters
                    if contains(model_name, 'KNN')
                        best_params = get_best_parameters(model_name, feature_type);
                        mdl = fitcknn(X_train_norm, y_train, ...
                            'NumNeighbors', best_params.K, ...
                            'Distance', 'euclidean', ...
                            'Standardize', false);
                    elseif contains(model_name, 'SVM-Linear')
                        best_params = get_best_parameters(model_name, feature_type);
                        mdl = fitcsvm(X_train_norm, y_train, ...
                            'KernelFunction', 'linear', ...
                            'BoxConstraint', best_params.C, ...
                            'ClassNames', [0 1], ...
                            'Standardize', false);
                    elseif contains(model_name, 'SVM-RBF')
                        best_params = get_best_parameters(model_name, feature_type);
                        mdl = fitcsvm(X_train_norm, y_train, ...
                            'KernelFunction', 'rbf', ...
                            'KernelScale', best_params.ks, ...
                            'BoxConstraint', best_params.C, ...
                            'ClassNames', [0 1], ...
                            'Standardize', false);
                    end
                    y_pred = predict(mdl, X_test_norm);
                end
                fold_time(fold) = toc / numel(y_test);
                
                % Calculate metrics
                [fold_acc(fold), fold_prec(fold), fold_rec(fold), fold_f1(fold)] = ...
                    calculate_comprehensive_metrics(y_test, y_pred);
                
            catch ME
                fprintf('Fold %d failed: %s\n', fold, ME.message);
                fold_acc(fold) = NaN;
                fold_prec(fold) = NaN;
                fold_rec(fold) = NaN;
                fold_f1(fold) = NaN;
                fold_time(fold) = NaN;
            end
        end
        
        % Calculate statistics
        results_cv(m).name = model_name;
        results_cv(m).mean_acc = nanmean(fold_acc);
        results_cv(m).std_acc = nanstd(fold_acc);
        results_cv(m).mean_prec = nanmean(fold_prec);
        results_cv(m).std_prec = nanstd(fold_prec);
        results_cv(m).mean_rec = nanmean(fold_rec);
        results_cv(m).std_rec = nanstd(fold_rec);
        results_cv(m).mean_f1 = nanmean(fold_f1);
        results_cv(m).std_f1 = nanstd(fold_f1);
        results_cv(m).mean_time = nanmean(fold_time);
        results_cv(m).fold_acc = fold_acc;
        
        fprintf('Mean: %.2f%% (±%.2f%%)', results_cv(m).mean_acc, results_cv(m).std_acc);
    end
    fprintf('\n');
end

function display_all_results(results_70, results_50, results_cv, has_50)
    % Display 70/30 results
    fprintf('\n%s\n', repmat('=', 100, 1));
    fprintf('RESULTS SUMMARY - 70/30 HOLDOUT\n');
    fprintf('%s\n', repmat('=', 100, 1));
    
    fprintf('%-20s | %6s | %5s | %5s | %5s | %4s | %4s | %4s | %4s | %8s\n', ...
        'Model', 'Acc', 'Prec', 'Rec', 'F1', 'TP', 'FP', 'TN', 'FN', 'Time(s)');
    fprintf('%s\n', repmat('-', 100, 1));
    
    for i = 1:numel(results_70)
        r = results_70(i);
        if ~isnan(r.accuracy)
            fprintf('%-20s | %5.2f%% | %.3f | %.3f | %.3f | %4d | %4d | %4d | %4d | %8.4f\n', ...
                r.name, r.accuracy, r.precision, r.recall, r.f1, ...
                r.tp, r.fp, r.tn, r.fn, r.time_per_sample);
        else
            fprintf('%-20s | %8s | %5s | %5s | %5s | %4s | %4s | %4s | %4s | %8s\n', ...
                r.name, 'FAILED', '-', '-', '-', '-', '-', '-', '-', '-');
        end
    end
    
    % Display 50/50 results if available
    if has_50 && ~isempty(results_50)
        fprintf('\n%s\n', repmat('=', 100, 1));
        fprintf('RESULTS SUMMARY - 50/50 HOLDOUT\n');
        fprintf('%s\n', repmat('=', 100, 1));
        
        for i = 1:numel(results_50)
            r = results_50(i);
            if ~isnan(r.accuracy)
                fprintf('%-20s | %5.2f%% | %.3f | %.3f | %.3f | %4d | %4d | %4d | %4d | %8.4f\n', ...
                    r.name, r.accuracy, r.precision, r.recall, r.f1, ...
                    r.tp, r.fp, r.tn, r.fn, r.time_per_sample);
            else
                fprintf('%-20s | %8s | %5s | %5s | %5s | %4s | %4s | %4s | %4s | %8s\n', ...
                    r.name, 'FAILED', '-', '-', '-', '-', '-', '-', '-', '-');
            end
        end
    end
    
    % Display Cross-Validation results
    fprintf('\n%s\n', repmat('=', 100, 1));
    fprintf('RESULTS SUMMARY - 5-FOLD CROSS VALIDATION\n');
    fprintf('%s\n', repmat('=', 100, 1));
    
    fprintf('%-20s | %12s | %12s | %12s | %12s | %8s\n', ...
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time(s)');
    fprintf('%s\n', repmat('-', 100, 1));
    
    for m = 1:numel(results_cv)
        r = results_cv(m);
        if ~isnan(r.mean_acc)
            fprintf('%-20s | %5.2f±%.2f%% | %5.3f±%.3f | %5.3f±%.3f | %5.3f±%.3f | %8.4f\n', ...
                r.name, r.mean_acc, r.std_acc, r.mean_prec, r.std_prec, ...
                r.mean_rec, r.std_rec, r.mean_f1, r.std_f1, r.mean_time);
        else
            fprintf('%-20s | %12s | %12s | %12s | %12s | %8s\n', ...
                r.name, 'FAILED', '-', '-', '-', '-');
        end
    end
end

function create_comprehensive_plots_all_methods(results_70, results_50, results_cv, has_50)
    % Create comparison figure
    fig = figure('Position', [100 100 1400 1000], 'Name', 'All Validation Methods Comparison');
    
    % Plot 1: Accuracy comparison across methods
    subplot(2, 3, 1);
    models = {results_70.name};
    acc_70 = [results_70.accuracy];
    
    if has_50 && ~isempty(results_50)
        acc_50 = [results_50.accuracy];
    else
        acc_50 = nan(size(acc_70));
    end
    acc_cv = [results_cv.mean_acc];
    
    x = 1:numel(models);
    hold on;
    plot(x, acc_70, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '70/30');
    if has_50 && ~isempty(results_50)
        plot(x, acc_50, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '50/50');
    end
    plot(x, acc_cv, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '5-Fold CV');
    
    set(gca, 'XTick', x, 'XTickLabel', models, 'XTickLabelRotation', 45);
    ylabel('Accuracy (%)');
    title('Accuracy Comparison - All Validation Methods');
    legend('show', 'Location', 'southeast');
    grid on;
    ylim([90, 100]);
    
    % Plot 2: Method reliability (CV standard deviation)
    subplot(2, 3, 2);
    cv_std = [results_cv.std_acc];
    bar(cv_std);
    set(gca, 'XTickLabel', models, 'XTickLabelRotation', 45);
    ylabel('Standard Deviation (%)');
    title('5-Fold CV Reliability (Lower = Better)');
    grid on;
    
    % Plot 3: Best model from each method
    subplot(2, 3, 3);
    [best_70_acc, best_70_idx] = max([results_70.accuracy]);
    best_70_name = results_70(best_70_idx).name;
    
    if has_50 && ~isempty(results_50)
        [best_50_acc, best_50_idx] = max([results_50.accuracy]);
        best_50_name = results_50(best_50_idx).name;
    else
        best_50_acc = NaN;
    end
    
    [best_cv_acc, best_cv_idx] = max([results_cv.mean_acc]);
    best_cv_name = results_cv(best_cv_idx).name;
    
    methods = {'70/30', '50/50', '5-Fold CV'};
    best_accs = [best_70_acc, best_50_acc, best_cv_acc];
    valid_best_accs = best_accs(~isnan(best_accs));
    valid_methods = methods(~isnan(best_accs));
    
    bar(valid_best_accs);
    set(gca, 'XTickLabel', valid_methods);
    ylabel('Best Accuracy (%)');
    title('Best Model from Each Validation Method');
    grid on;
    ylim([95, 100]);
    
    % Add accuracy values on bars
    for i = 1:numel(valid_best_accs)
        text(i, valid_best_accs(i) + 0.1, sprintf('%.2f%%', valid_best_accs(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    % Plot 4: Computational efficiency comparison
    subplot(2, 3, 4);
    times_70 = [results_70.time_per_sample] * 1000;
    times_cv = [results_cv.mean_time] * 1000;
    
    bar([times_70; times_cv]');
    set(gca, 'XTickLabel', models, 'XTickLabelRotation', 45);
    ylabel('Time per sample (ms)');
    title('Computational Efficiency Comparison');
    legend('70/30', '5-Fold CV', 'Location', 'northeast');
    grid on;
    
    % Plot 5: Feature type performance by method
    subplot(2, 3, 5);
    raw_idx = contains(models, 'RAW');
    hog_idx = contains(models, 'HOG');
    
    raw_acc_70 = mean(acc_70(raw_idx));
    hog_acc_70 = mean(acc_70(hog_idx));
    raw_acc_cv = mean(acc_cv(raw_idx));
    hog_acc_cv = mean(acc_cv(hog_idx));
    
    bar_data = [raw_acc_70, hog_acc_70; raw_acc_cv, hog_acc_cv];
    bar(bar_data);
    set(gca, 'XTickLabel', {'70/30', '5-Fold CV'});
    ylabel('Average Accuracy (%)');
    title('Feature Type Performance by Method');
    legend('RAW', 'HOG', 'Location', 'southeast');
    grid on;
    
    % Plot 6: Validation method recommendation
    subplot(2, 3, 6);
    method_scores = [mean(acc_70), mean(acc_50), mean(acc_cv)];
    method_stability = [std(acc_70), std(acc_50), nanstd(acc_cv)];
    
    % Simple scoring: higher accuracy, lower std is better
    scores = method_scores - method_stability;
    valid_scores = scores(~isnan(scores));
    valid_methods_plot = methods(~isnan(scores));
    
    bar(valid_scores);
    set(gca, 'XTickLabel', valid_methods_plot);
    ylabel('Performance Score');
    title('Recommended Validation Method (Higher = Better)'); % Fixed title - removed \\n
    grid on;
    
    % Add scores on bars
    for i = 1:numel(valid_scores)
        text(i, valid_scores(i) + 0.1, sprintf('%.1f', valid_scores(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    % Save figure
    if ~exist('results/figures', 'dir'), mkdir('results/figures'); end
    saveas(fig, 'results/figures/Fig_All_Validation_Methods_Comparison.png');
    fprintf('All validation methods comparison figure done\n');
end

function perform_comprehensive_statistical_analysis(results_70, results_50, results_cv, has_50)
    fprintf('\n%s\n', repmat('=', 70, 1));
    fprintf('COMPREHENSIVE STATISTICAL ANALYSIS & RECOMMENDATIONS\n');
    fprintf('%s\n', repmat('=', 70, 1));
    
    % Best models from each method
    [best_70_acc, best_70_idx] = max([results_70.accuracy]);
    best_70_name = results_70(best_70_idx).name;
    
    if has_50 && ~isempty(results_50)
        [best_50_acc, best_50_idx] = max([results_50.accuracy]);
        best_50_name = results_50(best_50_idx).name;
    else
        best_50_acc = NaN;
        best_50_name = 'N/A';
    end
    
    [best_cv_acc, best_cv_idx] = max([results_cv.mean_acc]);
    best_cv_name = results_cv(best_cv_idx).name;
    best_cv_std = results_cv(best_cv_idx).std_acc;
    
    fprintf('BEST MODELS:\n');
    fprintf('  70/30 Holdout:  %s (%.2f%%)\n', best_70_name, best_70_acc);
    if has_50 && ~isempty(results_50)
        fprintf('  50/50 Holdout:  %s (%.2f%%)\n', best_50_name, best_50_acc);
    end
    fprintf('  5-Fold CV:      %s (%.2f%% ± %.2f%%)\n', best_cv_name, best_cv_acc, best_cv_std);
    
    % Method comparison
    fprintf('\nMETHOD RELIABILITY:\n');
    cv_stds = [results_cv.std_acc];
    valid_cv_stds = cv_stds(~isnan(cv_stds));
    if ~isempty(valid_cv_stds)
        fprintf('  5-Fold CV average std: %.3f%%\n', mean(valid_cv_stds));
        [min_std, min_idx] = min(valid_cv_stds);
        fprintf('  Lowest CV std: %.3f%% (%s)\n', min_std, results_cv(min_idx).name);
        [max_std, max_idx] = max(valid_cv_stds);
        fprintf('  Highest CV std: %.3f%% (%s)\n', max_std, results_cv(max_idx).name);
    end
    
    % Feature analysis
    models = {results_70.name};
    raw_idx = contains(models, 'RAW');
    hog_idx = contains(models, 'HOG');
    
    raw_acc_70 = mean([results_70(raw_idx).accuracy]);
    hog_acc_70 = mean([results_70(hog_idx).accuracy]);
    
    fprintf('\nFEATURE PERFORMANCE (70/30):\n');
    fprintf('  RAW features: %.2f%%\n', raw_acc_70);
    fprintf('  HOG features: %.2f%%\n', hog_acc_70);
    
    if hog_acc_70 > raw_acc_70
        fprintf('  → HOG features perform better by %.2f%%\n', hog_acc_70 - raw_acc_70);
    else
        fprintf('  → RAW features perform better by %.2f%%\n', raw_acc_70 - hog_acc_70);
    end
    
    % Final recommendations
    fprintf('\nFINAL RECOMMENDATIONS:\n');
    fprintf('  1. Best overall model: %s\n', best_cv_name);
    fprintf('  2. Most reliable validation: 5-Fold CV\n');
    if hog_acc_70 > raw_acc_70
        fprintf('  3. Recommended features: HOG\n');
    else
        fprintf('  3. Recommended features: RAW\n');
    end
    fprintf('  4. For quick evaluation: 70/30 Holdout\n');
end

function [accuracy, precision, recall, f1, tp, fp, tn, fn] = calculate_comprehensive_metrics(y_true, y_pred)
    % Calculate all required metrics
    tp = sum((y_pred == 1) & (y_true == 1));
    fp = sum((y_pred == 1) & (y_true == 0));
    tn = sum((y_pred == 0) & (y_true == 0));
    fn = sum((y_pred == 0) & (y_true == 1));
    
    accuracy = (tp + tn) / (tp + fp + tn + fn) * 100;
    precision = tp / (tp + fp + eps);
    recall = tp / (tp + fn + eps);
    f1 = 2 * precision * recall / (precision + recall + eps);
end

function [mdl, mu, sigma] = load_model(model_name, feature_type)
    % Load appropriate model based on name and feature type
    if contains(model_name, 'NN')
        if strcmp(feature_type, 'hog')
            load('models/nn/modelNN_hog.mat', 'modelNN_hog');
            mdl = modelNN_hog;
            mu = modelNN_hog.mu;
            sigma = modelNN_hog.sigma;
        else
            load('models/nn/modelNN_raw.mat', 'modelNN_raw');
            mdl = modelNN_raw;
            mu = modelNN_raw.mu;
            sigma = modelNN_raw.sigma;
        end
        
    elseif contains(model_name, 'KNN')
        if strcmp(feature_type, 'hog')
            M = load('models/knn/modelKNN_hog.mat');
            mdl = M.modelKNN_hog;
            mu = M.mu_hog_full;
            sigma = M.sigma_hog_full;
        else
            M = load('models/knn/modelKNN_raw.mat');
            mdl = M.modelKNN_raw;
            mu = M.mu_raw_full;
            sigma = M.sigma_raw_full;
        end
        
    elseif contains(model_name, 'SVM-Linear')
        if strcmp(feature_type, 'hog')
            M = load('models/svm/modelSVM_linear_hog.mat');
            mdl = M.modelSVM_linear_hog;
            mu = M.mu_hog_full;
            sigma = M.sigma_hog_full;
        else
            M = load('models/svm/modelSVM_linear_raw.mat');
            mdl = M.modelSVM_linear_raw;
            mu = M.mu_raw_full;
            sigma = M.sigma_raw_full;
        end
        
    elseif contains(model_name, 'SVM-RBF')
        if strcmp(feature_type, 'hog')
            M = load('models/svm/modelSVM_rbf_hog.mat');
            mdl = M.modelSVM_hog;
            mu = M.mu_hog_full;
            sigma = M.sigma_hog_full;
        else
            M = load('models/svm/modelSVM_rbf_raw.mat');
            mdl = M.modelSVM_raw;
            mu = M.mu_raw_full;
            sigma = M.sigma_raw_full;
        end
    end
end

function y_pred = predict_nn(Xte_norm, model, model_name)
    % Custom prediction for NN models
    y_pred = zeros(size(Xte_norm, 1), 1);
    for i = 1:size(Xte_norm, 1)
        y_pred(i) = NNTesting(Xte_norm(i,:), model);
    end
end

function create_detailed_confusion_matrices(results, y_true, all_predictions, split_name)
    % Create confusion matrices for a specific validation method
    n_models = numel(results);
    
    % Determine subplot layout
    if n_models <= 4
        rows = 2; cols = 2;
    elseif n_models <= 6
        rows = 2; cols = 3;
    else
        rows = 3; cols = 3;
    end
    
    fig = figure('Position', [100 100 1400 1000], 'Name', ['Confusion Matrices - ' split_name], 'Visible', 'off');
    
    for i = 1:n_models
        subplot(rows, cols, i);
        
        if ~isnan(results(i).accuracy)
            cm = confusionmat(y_true, all_predictions(:, i));
            confusionchart(cm, {'Non-Pedestrian', 'Pedestrian'}, ...
                'Title', sprintf('%s\nAccuracy: %.2f%%', results(i).name, results(i).accuracy));
        else
            text(0.5, 0.5, 'Model Failed', 'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', 'red');
            title(sprintf('%s\nFAILED', results(i).name), 'Color', 'red');
            axis off;
        end
    end
    
    sgtitle(sprintf('Confusion Matrices - %s', split_name), 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    if ~exist('results/figures', 'dir'), mkdir('results/figures'); end
    saveas(fig, sprintf('results/figures/Fig_Confusion_Matrices_%s.png', strrep(split_name, '/', '_')));
    fprintf('✅ Confusion matrices saved for %s\n', split_name);
    close(fig);
end

function best_params = get_best_parameters(model_name, feature_type)
    % Get best parameters from pre-trained models
    best_params = struct();
    
    if contains(model_name, 'KNN')
        if strcmp(feature_type, 'hog')
            M = load('models/knn/modelKNN_hog.mat');
            best_params.K = M.bestHog.K;
        else
            M = load('models/knn/modelKNN_raw.mat');
            best_params.K = M.bestRaw.K;
        end
    elseif contains(model_name, 'SVM-Linear')
        if strcmp(feature_type, 'hog')
            M = load('models/svm/modelSVM_linear_hog.mat');
            best_params.C = M.bestHog.C;
        else
            M = load('models/svm/modelSVM_linear_raw.mat');
            best_params.C = M.bestRaw.C;
        end
    elseif contains(model_name, 'SVM-RBF')
        if strcmp(feature_type, 'hog')
            M = load('models/svm/modelSVM_rbf_hog.mat');
            best_params.C = M.bestHog.C;
            best_params.ks = M.bestHog.ks;
        else
            M = load('models/svm/modelSVM_rbf_raw.mat');
            best_params.C = M.bestRaw.C;
            best_params.ks = M.bestRaw.ks;
        end
    end
end