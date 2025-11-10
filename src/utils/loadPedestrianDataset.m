function [X, y, fileNames] = loadPedestrianDataset(posDir, negDir, featureType, targetSize, varargin)
% featureType: 'raw' | 'hog' | 'lbp' | 'edge' | 'edgehog' | 'fusion_hog_lbp'
% varargin for 'hog'/'edgehog': cellSize, blockSize, numBins
% varargin for 'lbp':          cellSize

    if nargin < 3 || isempty(featureType), featureType = 'raw'; end
    if nargin < 4 || isempty(targetSize),  targetSize  = [128 64]; end

    % defaults for HOG-like
    cellSize  = [8 8];
    blockSize = [2 2];
    numBins   = 9;

    % parse varargin per feature
    switch lower(featureType)
        case {'hog','edgehog'}
            if numel(varargin) >= 1 && ~isempty(varargin{1}), cellSize  = varargin{1}; end
            if numel(varargin) >= 2 && ~isempty(varargin{2}), blockSize = varargin{2}; end
            if numel(varargin) >= 3 && ~isempty(varargin{3}), numBins   = varargin{3}; end
        case 'lbp'
            if numel(varargin) >= 1 && ~isempty(varargin{1}), cellSize  = varargin{1}; end
        case {'edge','raw','fusion_hog_lbp'}
            % nothing special
    end

    % collect files
    posFiles = [dir(fullfile(posDir,'*.png')); dir(fullfile(posDir,'*.jpg')); dir(fullfile(posDir,'*.jpeg')); ...
                dir(fullfile(posDir,'*.PNG')); dir(fullfile(posDir,'*.JPG')); dir(fullfile(posDir,'*.JPEG'))];
    negFiles = [dir(fullfile(negDir,'*.png')); dir(fullfile(negDir,'*.jpg')); dir(fullfile(negDir,'*.jpeg')); ...
                dir(fullfile(negDir,'*.PNG')); dir(fullfile(negDir,'*.JPG')); dir(fullfile(negDir,'*.JPEG'))];

    Npos = numel(posFiles); Nneg = numel(negFiles);
    X = []; y = []; fileNames = cell(Npos+Nneg,1);

    % local helper to get one feature
    function f = getFeat(I)
        I = imresize(I, targetSize, 'bilinear');
        switch lower(featureType)
            case 'raw'
                f = extractFeature(I, 'raw');
            case 'hog'
                f = extractFeature(I, 'hog', cellSize, blockSize, numBins);
            case 'lbp'
                f = extractFeature(I, 'lbp', cellSize);
            case 'edge'
                f = extractFeature(I, 'edge', cellSize);
            case 'edgehog'
                f = extractFeature(I, 'edgehog', cellSize, blockSize, numBins);
            case 'fusion_hog_lbp'
                f1 = extractFeature(I, 'hog', cellSize, blockSize, numBins);
                f2 = extractFeature(I, 'lbp', cellSize);
                f  = [f1(:); f2(:)]';
                % L2 norm across concatenated vector
                n = norm(f);
                if n > 0, f = f / n; end
            otherwise
                error('Unknown featureType: %s', featureType);
        end
    end

    % POS
    for i = 1:Npos
        I = imread(fullfile(posDir, posFiles(i).name));
        if size(I,3)==3, I = rgb2gray(I); end
        f = getFeat(I);
        X = [X; f]; y = [y; 1]; fileNames{i} = posFiles(i).name; %#ok<AGROW>
    end

    % NEG
    for i = 1:Nneg
        I = imread(fullfile(negDir, negFiles(i).name));
        if size(I,3)==3, I = rgb2gray(I); end
        f = getFeat(I);
        X = [X; f]; y = [y; 0]; fileNames{Npos+i} = negFiles(i).name; %#ok<AGROW>
    end
end
