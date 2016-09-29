function [F, names, valNames] = ConvertNetwork(filename)
% ConvertNetwork Returns a struct array of factors from a HUGIN .net file
%   F contains the struct array of factors
%   names contains the variable names, so X_i has name names(i)
%   valNames contains the assignment names for each variable, so that
%     assignments{i}{1} is the first value taken by X_i as seen in SAMIAM
%
%   DO NOT MODIFY THIS FILE AS IT WILL CAUSE PROBLEMS WITH GRADING

% initialize variables
nameToIdx = struct(); % using the struct as an associative array
valNames = {};
varCard = []; % cardinality of variables
F = struct('var', {}, 'card', {}, 'val', {});

fid = fopen(filename, 'rt');
assert(fid ~= -1, ['Error opening file ', filename]);

tline = fgetl(fid);
while ischar(tline)    
  if strfind(tline, 'node ') == 1
    tmp = textscan(tline, '%*s %s');
    curName = char(tmp{1});
    nameToIdx.(curName) = numel(fieldnames(nameToIdx)) + 1;

    % find the state field
    tline = fgetl(fid);
    while tline(1) ~= '}'
      if ~isempty(strfind(tline, 'states'))
        startIdx = strfind(tline, '(') + 1;
        endIdx = strfind(tline, ')');
        assert(~isempty(endIdx), 'Error: state string spans multiple lines');
        endIdx = endIdx - 1;
        stateNames = tline(startIdx:endIdx);
        
        % tokenize
        curStateNames = {};
        while true
          [str stateNames] = strtok(stateNames);          
          if isempty(str), break; end          
          str = str(2:end-1); % remove quotes
          curStateNames{end+1} = str;          
        end
        valNames{end+1} = curStateNames;
        varCard(end+1) = numel(curStateNames);
        break;
      end      
      tline = fgetl(fid);
    end
    
    % chomp up remaining lines till we reach the closing brace
    tline = fgetl(fid);
    while tline(1) ~= '}'
      tline = fgetl(fid);
    end
    
  elseif strfind(tline, 'potential ') == 1
    % find the cpd members
    obIdx = strfind(tline, '(');
    cbIdx = strfind(tline, ')');
    barIdx = strfind(tline, '|');
    
    varName = strtrim(tline(obIdx+1:barIdx-1));    
    parents = strtrim(tline(barIdx+1:cbIdx-1));
    
    % tokenize the parents
    if (isempty(parents))
      allVars = cellstr(varName);
    else
      parentVars = {};
      while true
        [str parents] = strtok(parents);
        if isempty(str), break; end
        parentVars{end+1} = str;
      end
      allVars = parentVars;
      allVars{end+1} = varName;
    end
   
    % extract variable IDs and cardinalities
    varIds = zeros(1, numel(allVars));    
    for i = 1:numel(allVars)
      varIds(i) = nameToIdx.(allVars{i});
    end
    cards = varCard(varIds);
    
    % read in actual factor values
    tline = fgetl(fid);
    while isempty(strfind(tline, 'data '))
      tline = fgetl(fid);      
    end
    
    obIdx = strfind(tline, '(');
    
    valString = tline(obIdx+1:end);
    while isempty(strfind(tline, ';'))
      tline = fgetl(fid);
      valString = [valString tline];      
    end
    valString = strtrim(regexprep(valString, '[();]', ' '));
    
    tmp = textscan(valString, '%f', prod(cards));
    vals = tmp{1};
    
    % define the factor. note that because the values in the HUGIN format
    % are in row-major order, we have to reverse the order of the variables 
    % (and cardinalities!) so that the values are in the correct order
    F(end + 1) = struct('var', fliplr(varIds), 'card', fliplr(cards), 'val', vals(:)');
    
    % chomp up remaining lines till we reach the closing brace
    tline = fgetl(fid);
    while tline(1) ~= '}'
      tline = fgetl(fid);
    end    
  end  
  
  tline = fgetl(fid);
end

fclose(fid);
names = fieldnames(nameToIdx);

end

