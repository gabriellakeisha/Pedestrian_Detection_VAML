%% compare_all_validation_methods.m — Compare 70/30, 50/50, and CV
clearvars; close all; clc;

fprintf('   VALIDATION METHODS COMPARISON\n');

%% Load 70/30 results by evaluating saved models
results_70 = struct();
has_70 = false;

try
    % Load test data from 70/30 split
    S = load('splits/splits.mat');
    testIdx = S.testIdx;
    R = load('features/raw/features_raw.mat');
    H = load('features/hog/features_hog.mat');
    
    Xte_hog = double(H.X_hog(testIdx,:));
    yte = double(R.y(testIdx));
    
    fprintf('Loaded 70/30 split: %d train / %d test\n', ...
        numel(S.trainIdx), numel(testIdx));
    
    idx = 1;
    
    % KNN-HOG
    try
        MH = load('models/knn/modelKNN_hog.mat');
        Xte_hog_norm = (Xte_hog - MH.mu_hog_full) ./ MH.sigma_hog_full;
        pred_hog = predict(MH.modelKNN_hog, Xte_hog_norm);
        acc = mean(pred_hog==yte)*100;
        
        results_70(idx).name = 'KNN-HOG';
        results_70(idx).acc = acc;
        fprintf('KNN-HOG: %.2f%%\n', acc);
        idx = idx + 1;
    catch ME
        fprintf('KNN-HOG failed: %s\n', ME.message);
    end
    
    % SVM-Linear-HOG
    try
        MH = load('models/svm/modelSVM_linear_hog.mat');
        Xte_hog_norm = (Xte_hog - MH.mu_hog_full) ./ MH.sigma_hog_full;
        pred_hog = predict(MH.modelSVM_linear_hog, Xte_hog_norm);
        acc = mean(pred_hog==yte)*100;
        
        results_70(idx).name = 'SVM-Linear-HOG';
        results_70(idx).acc = acc;
        fprintf('SVM-Linear-HOG: %.2f%%\n', acc);
        idx = idx + 1;
    catch ME
        fprintf('SVM-Linear-HOG failed: %s\n', ME.message);
    end
    
    % SVM-RBF-HOG
    try
        MH = load('models/svm/modelSVM_rbf_hog.mat');
        Xte_hog_norm = (Xte_hog - MH.mu_hog_full) ./ MH.sigma_hog_full;
        pred_hog = predict(MH.modelSVM_hog, Xte_hog_norm);
        acc = mean(pred_hog==yte)*100;
        
        results_70(idx).name = 'SVM-RBF-HOG';
        results_70(idx).acc = acc;
        fprintf('SVM-RBF-HOG: %.2f%%\n', acc);
        idx = idx + 1;
    catch ME
        fprintf('SVM-RBF-HOG failed: %s\n', ME.message);
    end
    
    has_70 = (idx > 1);
    
catch ME
    fprintf('Could not compute 70/30 results: %s\n', ME.message);
    has_70 = false;
end

%% Load 50/50 results
try
    R50 = load('results/results_50_50.mat');
    fprintf('Loaded 50/50 results: %d models\n', numel(R50.results_50));
    has_50 = true;
catch
    fprintf('50/50 results not found - run b_eval_with_half_half.m first\n');
    has_50 = false;
end

%% Load CV results
try
    RCV = load('results/cv_results.mat');
    fprintf('Loaded CV results: %d models\n', numel(RCV.results));
    has_cv = true;
catch
    fprintf('CV results not found - run b_cross_validation.m first\n');
    has_cv = false;
end

if ~has_70 && ~has_50 && ~has_cv
    error('No results found! Run evaluation scripts first.');
end

%% Create comparison table
fprintf('   TABLE: VALIDATION METHOD COMPARISON\n');

fprintf('%-20s | %15s | %15s | %15s\n', ...
    'Model', '70/30 Hold-out', '50/50 Hold-out', '5-Fold CV');
fprintf('%s\n', repmat('-', 75, 1));

model_names = {'KNN-HOG', 'SVM-Linear-HOG', 'SVM-RBF-HOG'};

for i = 1:numel(model_names)
    model = model_names{i};
    
    % Get 70/30 result
    if has_70
        idx_70 = find(strcmp({results_70.name}, model));
        if ~isempty(idx_70)
            str_70 = sprintf('%.2f%%', results_70(idx_70).acc);
        else
            str_70 = 'N/A';
        end
    else
        str_70 = 'N/A';
    end
    
    % Get 50/50 result
    if has_50
        idx_50 = find(contains({R50.results_50.name}, model));
        if ~isempty(idx_50)
            str_50 = sprintf('%.2f%%', R50.results_50(idx_50).acc);
        else
            str_50 = 'N/A';
        end
    else
        str_50 = 'N/A';
    end
    
    % Get CV result
    if has_cv
        idx_cv = find(strcmp({RCV.results.name}, model));
        if ~isempty(idx_cv)
            str_cv = sprintf('%.2f±%.2f%%', ...
                RCV.results(idx_cv).mean_acc, RCV.results(idx_cv).std_acc);
        else
            str_cv = 'N/A';
        end
    else
        str_cv = 'N/A';
    end
    
    fprintf('%-20s | %15s | %15s | %15s\n', model, str_70, str_50, str_cv);
end

fprintf('\n');

%% Visualization
if has_50 || has_cv || has_70
    fig = figure('Position', [100 100 1200 500], ...
        'Name', 'Validation Methods Comparison');
    
    % Prepare data
    models = {'KNN-HOG', 'SVM-Linear-HOG', 'SVM-RBF-HOG'};
    data = zeros(3, 3);  % 3 models × 3 methods
    
    for i = 1:3
        % 70/30 results
        if has_70
            idx = find(strcmp({results_70.name}, models{i}));
            if ~isempty(idx)
                data(i,1) = results_70(idx).acc;
            end
        end
        
        % 50/50 results
        if has_50
            idx = find(contains({R50.results_50.name}, models{i}));
            if ~isempty(idx)
                data(i,2) = R50.results_50(idx).acc;
            end
        end
        
        % CV results
        if has_cv
            idx = find(strcmp({RCV.results.name}, models{i}));
            if ~isempty(idx)
                data(i,3) = RCV.results(idx).mean_acc;
            end
        end
    end
    
    % Bar chart
    b = bar(data);
    b(1).FaceColor = [0.2 0.6 0.8];  % 70/30
    b(2).FaceColor = [0.8 0.4 0.2];  % 50/50
    b(3).FaceColor = [0.2 0.8 0.4];  % CV
    
    set(gca, 'XTickLabel', models);
    xtickangle(45);
    ylabel('Accuracy (%)');
    title('Comparison of Validation Methods');
    legend('70/30 Hold-out', '50/50 Hold-out', '5-Fold CV', 'Location', 'southeast');
    grid on;
    
    % Dynamic y-axis based on actual data (fixes missing bar issue)
    min_val = min(data(data > 0));  % Get minimum non-zero value
    if ~isempty(min_val)
        ylim([floor(min_val - 1), 100]);
    else
        ylim([95 100]);
    end
    
    % Save
    if ~exist('results/figures', 'dir'), mkdir('results/figures'); end
    saveas(fig, 'results/figures/Fig_Validation_Methods_Comparison.png');
    fprintf('✅ Figure saved: results/figures/Fig_Validation_Methods_Comparison.png\n');
end

%% Analysis & Recommendations
fprintf('\n========================================\n');
fprintf('   ANALYSIS & RECOMMENDATIONS\n');
fprintf('========================================\n\n');

if has_70
    fprintf('70/30 Hold-out:\n');
    fprintf('  - Results: %.2f%% (best model)\n\n', max([results_70.acc]));
end

if has_50
    fprintf('50/50 Hold-out:\n');
    best_50 = max([R50.results_50.acc]);
    fprintf('  - Results: %.2f%% (best model)\n', best_50);
    if has_70
        fprintf('  - Difference from 70/30: %.2f%%\n', ...
            best_50 - max([results_70.acc]));
    end
    fprintf('\n');
end

if has_cv
    fprintf('5-Fold Cross-Validation:\n');
    [~, best_cv_idx] = max([RCV.results.mean_acc]);
    fprintf('  - Results: %.2f±%.2f%% (best model)\n', ...
        RCV.results(best_cv_idx).mean_acc, RCV.results(best_cv_idx).std_acc);
    fprintf('  - Low std dev (±%.2f%%) indicates stable performance\n\n', ...
        RCV.results(best_cv_idx).std_acc);
end