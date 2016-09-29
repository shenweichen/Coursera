% IndexToAssignment Convert index to variable assignment.
%
%   A = IndexToAssignment(I, D) converts an index, I, into the .val vector
%   into an assignment over variables with cardinality D. If I is a vector, 
%   then the function produces a matrix of assignments, one assignment 
%   per row.
%
%   See also AssignmentToIndex.m and FactorTutorial.m

function A = IndexToAssignment(I, D)

D = D(:)'; % ensure that D is a row vector
A = mod(floor(repmat(I(:) - 1, 1, length(D)) ./ repmat(cumprod([1, D(1:end - 1)]), length(I), 1)), ...
        repmat(D, length(I), 1)) + 1;
  
end
