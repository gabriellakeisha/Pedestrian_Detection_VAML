function f = extractFeature(I, featureType, cellSize, blockSize, numBins)
% Unified feature extractor
% featureType: 'raw' | 'hog' | 'lbp' | 'edge' | 'edgehog'
% For 'hog'/'edgehog' you can pass cellSize, blockSize, numBins
% For 'lbp' you can pass cellSize (LBP cell) as 3rd arg

    if nargin < 2 || isempty(featureType), featureType = 'raw'; end
    if nargin < 3 || isempty(cellSize),    cellSize  = [8 8];  end
    if nargin < 4 || isempty(blockSize),   blockSize = [2 2];  end
    if nargin < 5 || isempty(numBins),     numBins   = 9;      end

    % grayscale single
    if size(I,3) == 3, I = rgb2gray(I); end
    I = im2single(I);

    switch lower(featureType)
        case 'raw'
            f = I(:)';

        case 'hog'
            f = extractHOGFeatures(I, ...
                'CellSize', cellSize, ...
                'BlockSize', blockSize, ...
                'NumBins',   numBins);

        case 'lbp'
            % LBP over cells, then L2-normalise
            f = extractLBPFeatures(I, 'CellSize', cellSize);
            n = norm(f);
            if n > 0, f = f / n; end

        case 'edge'
            % Flattened Canny edge map + density per cell
            BW = edge(I, 'Canny');
            % Downsample into same grid as cellSize for rough histogram
            cs = cellSize;
            [H,W] = size(BW);
            ny = floor(H/cs(1)); nx = floor(W/cs(2));
            if ny < 1 || nx < 1
                f = BW(:)';  % fallback
            else
                BWr = BW(1:ny*cs(1), 1:nx*cs(2));
                BWr = reshape(BWr, cs(1), ny, cs(2), nx);
                BWr = permute(BWr, [1 3 2 4]);
                cellCounts = squeeze(sum(sum(BWr,1),2)); % ny x nx
                f = [BW(:)'  double(cellCounts(:))'];
                n = norm(f);
                if n > 0, f = f / n; end
            end

        case 'edgehog'
            % HOG on Canny edge image: robust "edge/gradient" descriptor
            BW = edge(I, 'Canny');
            f = extractHOGFeatures(single(BW), ...
                'CellSize', cellSize, ...
                'BlockSize', blockSize, ...
                'NumBins',   numBins);

        otherwise
            error('Unknown featureType: %s', featureType);
    end
end
