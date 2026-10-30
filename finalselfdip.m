%% Intensity Transformation (Point Processing) with Evaluation Metrics
%  Techniques implemented (Chapter 3, Gonzalez & Woods):
%    1. Contrast stretching            (Fig. 3.10)
%    2. Intensity-level slicing        (Fig. 3.11)
%    3. Power-law (gamma) transform    (Eq. 3-5, s = c*r^gamma)
%    4. Image negative                 (Eq. 3-3, s = L-1-r)
%
%  Metrics: Mean, Std (contrast), Entropy, Dynamic range, MSE, PSNR, SSIM

clear; clc; close all;

%% 1. LOAD IMAGE
imgPath = 'cameraman.tif';     % built-in image. For your own file use e.g.
                               % imgPath = 'myphoto.png';
makeLowContrast = true;        % true = squeeze the image into a narrow range so
                               % that contrast stretching has something to fix

[raw, map] = imread(imgPath);
if ~isempty(map),           raw = ind2gray(raw, map);        end   % indexed PNG
if size(raw,3) >= 3,        raw = rgb2gray(raw(:,:,1:3));    end   % RGB/RGBA
if isa(raw,'uint16'),       raw = uint8(double(raw)/257);    end   % 16-bit
f = uint8(raw);

if makeLowContrast
    f = uint8(80 + double(f) * (80/255));      % range becomes about [80,160]
end

L = 256;                                        % number of gray levels
fprintf('Image size: %d x %d,  intensity range [%d, %d]\n', ...
        size(f,1), size(f,2), min(f(:)), max(f(:)));

%% 2. DEFINE THE FOUR TRANSFORMATIONS  s = T(r)
% (1) Contrast stretching: (r1,s1) = (rmin,0), (r2,s2) = (rmax,L-1)
rmin = double(min(f(:)));
rmax = double(max(f(:)));
if rmax == rmin, rmax = rmin + 1; end
T1 = @(x) (double(x) - rmin) / (rmax - rmin) * (L-1);

% (2) Intensity-level slicing (binary version, Fig. 3.11(a))
%     range of interest [A, B] -> white, everything else -> black
mu = mean(double(f(:)));   sd = std(double(f(:)));
A  = round(mu - 0.5*sd);                       % change A and B as you like
B  = round(mu + 0.5*sd);
T2 = @(x) (L-1) * double(double(x) >= A & double(x) <= B);

% (3) Power-law (gamma) transformation
c   = 1;
gam = 0.5;                                     % <1 brightens, >1 darkens
T3  = @(x) (L-1) * c * (double(x)/(L-1)).^gam;

% (4) Image negative
T4  = @(x) (L-1) - double(x);

names = {'Original', 'Contrast stretching', 'Intensity slicing', ...
         sprintf('Gamma (%.2f)', gam), 'Negative'};

%% 3. APPLY TRANSFORMATIONS
g = cell(1,5);
g{1} = f;
g{2} = uint8(T1(f));
g{3} = uint8(T2(f));
g{4} = uint8(T3(f));
g{5} = uint8(T4(f));

%% 4. EVALUATION METRICS
fprintf('\nSlicing range: A = %d, B = %d\n', A, B);
fprintf('\n%-22s %8s %8s %8s %8s %10s %9s %8s\n', ...
    'Method','Mean','Std','Entropy','Range','MSE','PSNR(dB)','SSIM');
fprintf('%s\n', repmat('-', 1, 95));

M = zeros(5, 7);
for k = 1:5
    M(k,:) = compute_metrics(f, g{k});
    fprintf('%-22s %8.2f %8.2f %8.3f %8d %10.2f %9.2f %8.4f\n', ...
        names{k}, M(k,1), M(k,2), M(k,3), M(k,4), M(k,5), M(k,6), M(k,7));
end

fprintf(['\nHow to read the table:\n' ...
 '  Std       : global contrast (higher = more contrast)\n' ...
 '  Entropy   : amount of information/detail (higher = richer)\n' ...
 '  Range     : max - min intensity (dynamic range used)\n' ...
 '  MSE/PSNR/SSIM compare the output with the original; for enhancement\n' ...
 '  they measure HOW MUCH the image changed, not quality by themselves.\n']);

%% 5. DISPLAY: images + histograms
figure('Name','Intensity transformations','NumberTitle','off', ...
       'Position',[50 50 1400 600]);
for k = 1:5
    subplot(2,5,k);
    imshow(g{k}, [0 255]);
    title(names{k});

    subplot(2,5,5+k);
    h = histcounts(double(g{k}(:)), 0:256);
    bar(0:255, h, 1, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'none');
    xlim([0 255]);
    xlabel('Intensity'); ylabel('Pixels');
end

%% 6. DISPLAY: transformation curves s = T(r)
r = 0:255;
figure('Name','Transformation functions','NumberTitle','off');
plot(r, T1(r), 'LineWidth', 1.5); hold on;
plot(r, T2(r), 'LineWidth', 1.5);
plot(r, T3(r), 'LineWidth', 1.5);
plot(r, T4(r), 'LineWidth', 1.5);
plot(r, r, 'k:');
hold off; grid on; axis([0 255 0 255]); axis square;
xlabel('Input intensity, r'); ylabel('Output intensity, s');
legend('Contrast stretching','Slicing','Gamma','Negative','Identity', ...
       'Location','southoutside');
title('Intensity transformation functions');

%% 7. BAR CHART OF KEY METRICS
figure('Name','Metrics comparison','NumberTitle','off');
subplot(1,3,1); bar(M(:,2)); title('Std (contrast)');
set(gca,'XTickLabel',names,'XTickLabelRotation',30);
subplot(1,3,2); bar(M(:,3)); title('Entropy (bits)');
set(gca,'XTickLabel',names,'XTickLabelRotation',30);
subplot(1,3,3); bar(M(:,7)); title('SSIM vs original');
set(gca,'XTickLabel',names,'XTickLabelRotation',30);

%% ============================================================
%  LOCAL FUNCTION: image quality / enhancement metrics
%% ============================================================
function m = compute_metrics(ref, out)
    refd = double(ref);
    outd = double(out);

    meanVal = mean(outd(:));
    stdVal  = std(outd(:));

    p = histcounts(outd(:), 0:256) / numel(outd);
    p = p(p > 0);
    ent = -sum(p .* log2(p));

    rng = max(outd(:)) - min(outd(:));

    mseVal = mean((refd(:) - outd(:)).^2);
    if mseVal == 0
        psnrVal = Inf;
    else
        psnrVal = 10 * log10(255^2 / mseVal);
    end

    try
        ssimVal = ssim(uint8(out), uint8(ref));   % Image Processing Toolbox
    catch
        ssimVal = NaN;
    end

    m = [meanVal, stdVal, ent, rng, mseVal, psnrVal, ssimVal];
end