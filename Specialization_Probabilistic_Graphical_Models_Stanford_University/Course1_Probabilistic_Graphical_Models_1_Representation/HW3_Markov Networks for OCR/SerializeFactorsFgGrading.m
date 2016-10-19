function out = SerializeFactorsFgGrading(F, skip)
% Serializes the factors similar to SerializeFactorsFg
% but drops factors with values = 1
% This is only used during grading.

if nargin == 1
  skip = 1;
end

lines = cell(5*numel(F) + 1, 1);

lines{1} = sprintf('%d\n', numel(F));
lineIdx = 2;
for i = 1:numel(F)
  lines{lineIdx} = sprintf('\n%d\n', numel(F(i).var));
  lineIdx = lineIdx + 1;

  lines{lineIdx} = sprintf('%s\n', num2str(F(i).var(:)')); % ensure that we put in a row vector
  lineIdx = lineIdx + 1;

  lines{lineIdx} = sprintf('%s\n', num2str(F(i).card(:)')); % ensure that we put in a row vector
  lineIdx = lineIdx + 1;

  lines{lineIdx} = sprintf('%d\n', numel(F(i).val));
  lineIdx = lineIdx + 1;

  % Internal storage of factor vals is already in the same indexing order
  % as what libDAI expects, so we do not need to convert the indices.
  allVals = F(i).val(:);
  selIdx = (allVals ~= 1);
  vals = [(allVals(selIdx))'];

  % selIdx = (F(i).val(:) ~= 1);
  % vals = [(F(i).val(:)(selIdx))'];

  vals = vals(1:skip:numel(vals));
  lines{lineIdx} = sprintf('%0.8g\n', vals);
  lineIdx = lineIdx + 1;
end

out = sprintf('%s', lines{:});
end
