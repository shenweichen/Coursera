function factors = ChooseTopSimilarityFactors (allFactors, F)
% This function chooses the similarity factors with the highest similarity
% out of all the possibilities.
%
% Input:
%   allFactors: An array of all the similarity factors.
%   F: The number of factors to select.
%
% Output:
%   factors: The F factors out of allFactors for which the similarity score
%     is highest.
%
% Hint: Recall that the similarity score for two images will be in every
%   factor table entry (for those two images' factor) where they are
%   assigned the same character value.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

% If there are fewer than F factors total, just return all of them.
if (length(allFactors) <= F)
    factors = allFactors;
    return;
end

% Your code here:
%factors = allFactors; %%% REMOVE THIS LINE
for i = 1:length(allFactors)
    allFactors(i).key = allFactors(i).val(1);
end
[a,idx] = sort([allFactors.key],'descend')
factors = allFactors(idx(1:F))
factors = rmfield(factors,'key')
%factors = factors(1:F)
end

