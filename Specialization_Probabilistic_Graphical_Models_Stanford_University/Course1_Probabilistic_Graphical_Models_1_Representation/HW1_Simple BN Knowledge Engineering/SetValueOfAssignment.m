% SetValueOfAssignment Sets the value of a variable assignment in a factor.
%
%   F = SetValueOfAssignment(F, A, v) sets the value of a variable assignment,
%   A, in factor F to v. The order of the variables in A are assumed to be the
%   same as the order in F.var.
%
%   F = SetValueOfAssignment(F, A, v, VO) sets the value of a variable
%   assignment, A, in factor F to v. The order of the variables in A are given
%   by the vector VO.
%
%   Note that SetValueOfAssignment *does not modify* the factor F that is 
%   passed into the function, but instead returns a modified factor with the 
%   new value(s) for the specified assignment(s). This is why we have to  
%   reassign F to the result of SetValueOfAssignment in the code snippets 
%   shown above.
%
%   See also GetValueOfAssignment.m and FactorTutorial.m

function F = SetValueOfAssignment(F, A, v, VO)

if (nargin == 3),
    indx = AssignmentToIndex(A, F.card);
else
    map = zeros(length(F.var), 1);
    for i = 1:length(F.var),
        map(i) = find(VO == F.var(i));
    end;
    indx = AssignmentToIndex(A(map), F.card);
end;

F.val(indx) = v;

end
