% Detailed description of the factor data structure and related functions
% -----------------------------------------------------------------------
% We will use structures to implement the factor datatype. The code
%
%   phi = struct('var', [3 1 2], 'card', [2 2 2], 'val', ones(1, 8));
%
% creates a factor over variables X_3, X_1, X_2, which are all binary
% valued, because phi.card(1) (the cardinality of X_3, |Val(X_3)|) is 2, 
% and likewise for X_1 and X_2. phi has been initialized so that 
% phi(X_3, X_1, X_2) = 1 for any assignment to the variables.
%
% A factor's values are stored in a row vector in the .val field 
% using an ordering such that the left-most variables as defined in the 
% .var field cycle through their values the fastest. More concretely, for 
% the factor phi defined above, we have the following mapping from variable 
% assignments to the index of the row vector in the .val field:
%
% -+-----+-----+-----+-------------------+   
%  | X_3 | X_1 | X_2 | phi(X_3, X_1, X_2)|
% -+-----+-----+-----+-------------------+
%  |  1  |  1  |  1  |     phi.val(1)    |
% -+-----+-----+-----+-------------------+
%  |  2  |  1  |  1  |     phi.val(2)    |
% -+-----+-----+-----+-------------------+
%  |  1  |  2  |  1  |     phi.val(3)    |
% -+-----+-----+-----+-------------------+
%  |  2  |  2  |  1  |     phi.val(4)    |
% -+-----+-----+-----+-------------------+
%  |  1  |  1  |  2  |     phi.val(5)    |
% -+-----+-----+-----+-------------------+
%  |  2  |  1  |  2  |     phi.val(6)    |
% -+-----+-----+-----+-------------------+
%  |  1  |  2  |  2  |     phi.val(7)    |
% -+-----+-----+-----+-------------------+
%  |  2  |  2  |  2  |     phi.val(8)    |
% -+-----+-----+-----+-------------------+
%
%
% We have provided the AssignmentToIndex and IndexToAssignment functions
% that compute the mapping between the assignments A and the variable indices I,
% given D, the cardinality of the variables. Concretely, given a factor phi, if
% phi.val(I) corresponds to the assignment A, i.e. phi(X = A) = phi.val(I) then
% 
%   I = AssignmentToIndex(A, D)
%   A = IndexToAssignment(I, D)
%
% For instance, for the factor phi as defined above, with the assignment 
%
%    A = [2 1 2] 
%
% to X_3, X_1 and X_2 respectively (as defined by phi.var = [3 1 2]), I = 6 
% as phi.val(6) corresponds to the value of phi(X_3 = 2, X_1 = 1, X_2 = 2).
% Thus, AssignmentToIndex([2 1 2], [2 2 2]) returns 6, and conversely, 
% IndexToAssignment(6, [2 2 2]) returns the vector [2 1 2]. The second
% argument in the function calls corresponds to the cardinality of the
% sample factor phi, phi.card, which is [2 2 2].
%
% More generally, the assignment vector A is a row vector that corresponds
% to assignments to the variables in a factor, with an understanding that the
% variables for which the assignments refer to are given by the .var field
% of the factor. 
%
% Giving AssignmentToIndex a matrix A, one assignment per row, will cause it
% to return a vector of indices I, such that I(k) is the index
% corresponding to the assignment in A(k, :) (row k). 
% 
% Similarly, giving IndexToAssignment a vector I of indices will yield a
% matrix A of assignments, one per row, such that A(k, :) (the kth row of A)
% corresponds to the assignment mapped to by index I(k).
%
% Getting and setting values to factors
% -------------------------------------
%
% We have provided the convenience functions GetValueOfAssignment and
% SetValueOfAssignment so you do not need to manipulate the .val field
% directly for getting and setting values.
%
% For instance, calling 
%
%   GetValueOfAssignment(phi, [1 2 1]) 
%
% yields the value phi(X_3 = 1, X_1 = 2, X_2 = 1). Again, the variables for 
% which the assignments refer to are given by the .var field of the factor.
%
% Similarly, executing 
%
%    phi = SetValueOfAssignment(phi, [2 2 1], 6) 
%
% causes the value of phi(X_3 = 2, X_1 = 2, X_2 = 1) to be set to 6. Note 
% that because MATLAB/Octave passes function arguments by value (not by 
% reference), SetValueOfAssignment *does not modify* the factor that you 
% passed in. Instead, it returns a new factor with the modified value for 
% the specified assignment; this is why we reassigned phi to the result of
% SetValueOfAssignment.
%
% More details about these functions are provided in their respective .m
% files.

% Sample Factors and Outputs
% --------------------------
%
% In the following, we have provided you with some sample input factors, as 
% well as the output that you should receive when various operations are
% performed on these factors. You may find these sample inputs and outputs
% helpful in debugging your implementation. For instance, FACTORS.PRODUCT 
% is the factor you should get when you execute 
%
%   FactorProduct(FACTORS.INPUT(1), FACTORS.INPUT(2))
%
% These sample factors define a simple chain Bayesian network over binary 
% variables: X_1 -> X_2 -> X_3
%

% FACTORS.INPUT(1) contains P(X_1)
FACTORS.INPUT(1) = struct('var', [1], 'card', [2], 'val', [0.11, 0.89]);

% FACTORS.INPUT(2) contains P(X_2 | X_1)
FACTORS.INPUT(2) = struct('var', [2, 1], 'card', [2, 2], 'val', [0.59, 0.41, 0.22, 0.78]);

% FACTORS.INPUT(3) contains P(X_3 | X_2)
FACTORS.INPUT(3) = struct('var', [3, 2], 'card', [2, 2], 'val', [0.39, 0.61, 0.06, 0.94]);

% Factor Product
%FACTORS.PRODUCT = FactorProduct(FACTORS.INPUT(1), FACTORS.INPUT(2));
% The factor defined here is correct to 4 decimal places.
FACTORS.PRODUCT = struct('var', [1, 2], 'card', [2, 2], 'val', [0.0649, 0.1958, 0.0451, 0.6942]);

% Factor Marginalization
% FACTORS.MARGINALIZATION = FactorMarginalization(FACTORS.INPUT(2), [2]);
FACTORS.MARGINALIZATION = struct('var', [1], 'card', [2], 'val', [1 1]); 

% Observe Evidence
% FACTORS.EVIDENCE = ObserveEvidence(FACTORS.INPUT, [2 1; 3 2]);
FACTORS.EVIDENCE(1) = struct('var', [1], 'card', [2], 'val', [0.11, 0.89]);
FACTORS.EVIDENCE(2) = struct('var', [2, 1], 'card', [2, 2], 'val', [0.59, 0, 0.22, 0]);
FACTORS.EVIDENCE(3) = struct('var', [3, 2], 'card', [2, 2], 'val', [0, 0.61, 0, 0]);

% Compute Joint Distribution
% FACTORS.JOINT = ComputeJointDistribution(FACTORS.INPUT);
FACTORS.JOINT = struct('var', [1, 2, 3], 'card', [2, 2, 2], 'val', [0.025311, 0.076362, 0.002706, 0.041652, 0.039589, 0.119438, 0.042394, 0.652548]);

% Compute Marginal
%FACTORS.MARGINAL = ComputeMarginal([2, 3], FACTORS.INPUT, [1, 2]);
FACTORS.MARGINAL = struct('var', [2, 3], 'card', [2, 2], 'val', [0.0858, 0.0468, 0.1342, 0.7332]);
