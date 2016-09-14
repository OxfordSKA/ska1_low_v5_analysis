function [final_index_list, mindist] = optimize_cables(antenna_xy, cluster_xy,...
    num_in_gen, num_survivors, num_generations, num_mutations)
% Genetic algorithm for finding cluster station matches.
%
% antenna_xy is (2, num_antennas)
% cluster_xy is (2, num_clusters)
%
%
% Args:
%   antenna_xy (array): antenna / station positions (2, na=162) 
%   cluster_xy (array): cluster centre positions (2, np=27)
%   num_in_gen (int): Number in the generation. (use ~1000)
%   num_survivors (int): Number of survivors per generation (use ~3)
%   num_generations (int): Number of generations (use ~200)
%   num_mutations (int): Number of mutations (use ~1)
% 
% Returns:
%   array: best (workspace variable) used for reindexing antennas (a) for
%   the given solution. ie a_new = a(:, best)
%
%
num_centres = int32(size(cluster_xy,2));   % Number of cluster centres
num_stations = int32(size(antenna_xy,2));  % Number of stations

num_in_gen = int32(num_in_gen);        % Number in a generation
num_survivors = int32(num_survivors);  % Number of survivors
num_generations = int32(num_generations);   
num_mutations = int32(num_mutations);

num_in_gen = num_in_gen + 1;
if (mod(num_stations, num_centres) ~= 0)
    disp ('antenna and points data are not compatible')
    return
end

seq = zeros(num_stations, num_in_gen, 'int32');  % Station index list for each child in generation.
cable_lengths = zeros(num_survivors, 1);  % Cable length for each of the survivors
index_list = zeros(num_stations, num_survivors, 'int32');  % Station index list for each survivor
cluster_index = zeros(num_stations, 1, 'int32');  % Cluster index of each antenna

% 1. Create the first generation of survivors (original parents)

% Get cable length for survivor0, which indexes stations in original order
index_list(:, 1) = int32(1:num_stations); 
[cable_lengths(1), dd, cluster_index(:,1)] = getdist(antenna_xy, cluster_xy);

% Get cable length for survivors 1 to n (different ordering)
for j = 2:num_survivors
    index_list(:, j) = index_list(:, 1);  % Initialise with original ordering
    cluster_index(:, j) = cluster_index(:, 1);  % Initialise cluster association
end

% Loop over all survivors 
for j = 1:num_survivors
    cable_length_ = 0;
    for i = 1:num_stations;
        cable_length_ = cable_length_ + dd(index_list(i,j), cluster_index(i,1));
    end
    cable_lengths(j) = cable_length_;
end

% [cable_lengths, ind] = sort(cable_lengths, 'ascend');
% index_list(:, :) = index_list(:, ind);
% cluster_index = cluster_index(:, ind);
% mindist = cable_lengths(1);

%
% Loop over the generations
%

nq = (num_in_gen - 1) / num_survivors;    % 333
nr = mod(num_in_gen - 1, num_survivors);  % 0

% Loop over generations
for jgen = 2:num_generations
    seq(:, 1) = index_list(:, 1);
    
    % Mutations (loop over parents from previous gen and mutate)
    i2 = 1;
    for j = 1:num_survivors
        i1 = i2 + 1;
        i2 = j * nq + min(j, nr) + 1;  % j * 333 + 0 + 1 (j == 1..3)
        
        for i = i1:i2  % i == child index (looping in blocks of children for each parent/surv)
            seq(:, i) = index_list(:, j);  % Init child station indices to that of parent
            % fseq(i) = cable_lengths(j);    % Init cable length of child
            
            % Apply mutations to each child
            for k=1:num_mutations
                % Station ids to swap.
                ia = randi(num_stations, 1);
                ib = randi(num_stations, 1);
                
                % Swap if different.
                if (ia ~= ib)
                    % Station 'ia' in child list i ==> parent antenna ib
                    seq(ia, i) = index_list(ib, j);
                    % Station 'ib' in child list i ==> parent antenna ia
                    seq(ib, i) = index_list(ia, j);
                    
                    % station ids after the swap
                    iia = seq(ia, i);
                    iib = seq(ib, i);
                   
                    % Cluster ids of the stations involved in the swap.
                    jza = cluster_index(ia, 1);  
                    jzb = cluster_index(ib, 1);
                    
                    % dd(station_id, cluster_id) == cluster - station cable length
                    % add1 is the extra cable length after the swap and
                    % add2 is the cable length before the swap
                    add1 = +dd(iib, jzb) + dd(iia, jza);
                    add2 = -dd(iib, jza) - dd(iia, jzb);
                    
                    % Work out new cable length by modifying the parent
                    % cable length,
                    fseqi = cable_lengths(j) + add1 + add2;
                    
                    % if the cable length of the child is less than the
                    % worst surivor, replace the worse survivor
                    if fseqi < cable_lengths(num_survivors)
                        % Update the saved cable length of the worse survivor
                        cable_lengths(num_survivors) = fseqi;
                        % Update the antenna index list for the worse
                        % survivor
                        index_list(:, num_survivors) = seq(:, i);
                        % Sort the cable lengths of survivors
                        [cable_lengths, ind] = sort(cable_lengths, 'ascend');
                        % Sort the index list by cable length
                        index_list(:, :) = index_list(:, ind);
                    end
                end
            end
        end
    end
    
    % Assume survivor of the last generation with the shortest cable length
    % is the indices we want.
    final_index_list = index_list(:, 1);
    mindist = cable_lengths(1);
    
end

