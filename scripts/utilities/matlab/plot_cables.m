function [dist] = plot_cables(a, p )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

hold on
j2 = 0;
dist = 0;
for i = 1:27
    j1 = j2+1;
    j2 = j2+6;
    plot(p(1,i), p(2,i), 'r+')    
    plot(a(1,j1:j2), a(2,j1:j2),'b.')
    for j = j1:j2
        plot([p(1,i); a(1, j)], [p(2,i);a(2,j)], 'k-')
        dist = dist + sqrt((a(1, j) - p(1, i))^2 + (a(2, j) - p(2, i))^2);
    end
end
hold off
end

