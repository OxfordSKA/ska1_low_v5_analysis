function [d, dd, jdd] = getdist(antenna_xy, cluster_xy)
% Work out the total cable length.
% 
% Returns:
%  d = total cable length (in m)
%  dd = matrix of distances from each cluster centre to each antenna
%  jdd = cluster index of each antenna
%  

num_clusters = size(cluster_xy, 2);
num_antennas = size(antenna_xy, 2);
num_clusters = int32(num_clusters);
num_per_cluster = int32(num_antennas / num_clusters);  % == 6 number of antennas per cluster

dd = zeros(num_antennas,num_clusters);
dxy = zeros(2,num_antennas);

% Loop over cluster centres
for j=1:num_clusters
    dxy(1, :) = antenna_xy(1, :) - cluster_xy(1, j);  % == dx   
    dxy(2, :) = antenna_xy(2, :) - cluster_xy(2, j);  % == dy   
    dd(:,j) = sqrt(sum(dxy.^2, 1));  % dr for all antennas (:) for cluster j
end


% Get total cable length which is sum of a subset of dd because dd is the 
% distance of all antennas to all centres.
d = 0;
i2 = 0;
for j=1:num_clusters
    i1 = i2 + 1;
    i2 = i2 + num_per_cluster;
    jdd(i1:i2) = j;
    d = d + sum(dd(i1:i2, j));
end

end

