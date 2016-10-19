% IndexToAssignment Convert index to variable assignment.
%
%   A = IndexToAssignment(I, D) converts an index, I, into the .val vector
%   into an assignment over variables with cardinality D. If I is a vector, 
%   then the function produces a matrix of assignments, one assignment 
%   per row.
%
%   See also AssignmentToIndex.m
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function A = IndexToAssignment(I, D)

D = D(:)'; % ensure that D is a row vector
A = bsxfun(@mod, floor(bsxfun(@rdivide, I(:) - 1, cumprod([1, D(1:end - 1)]))), D) + 1;
  
end
