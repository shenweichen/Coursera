function out = SerializeFactorsFg(F)
% Serializes a factor struct array into the .fg format for libDAI
% http://cs.ru.nl/~jorism/libDAI/doc/fileformats.html
%
% To avoid incompatibilities with EOL markers, make sure you write the
% string to a file using the appropriate file type ('wt' for windows, 'w'
% for unix)
%
% Copyright (C) Daphne Koller, Stanford University, 2012


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
  % as what libDAI expects, so we don't need to convert the indices.
  vals = [0:(numel(F(i).val) - 1); F(i).val(:)'];
  lines{lineIdx} = sprintf('%d %0.8g\n', vals);
  lineIdx = lineIdx + 1;
end

out = sprintf('%s', lines{:});
end
