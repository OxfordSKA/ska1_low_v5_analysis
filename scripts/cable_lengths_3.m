addpath('utilities/matlab/');

% Settings 
nseq = 1000;
nsurv = 3;
ngen = 500;
nmut = 1;

% results_dir = './TEMP_results';
% files = dir(results_dir);


% clear a p antennas_x antennas_y centre_x centre_y sort_idx;
load 'TEMP_results/model02_r08_layout.mat';
a = [antennas_x(:)'; antennas_y(:)'];
p = [centre_x; centre_y];
% dist = plot_cables(a, p);
[sort_idx, min_dist] = optimize_cables(a, p, nseq, nsurv, ngen, nmut);
fprintf('min_dist = %.2f\n', min_dist/1e3)
dist = plot_cables(a(:, sort_idx), p);

% clear a p antennas_x antennas_y centre_x centre_y;