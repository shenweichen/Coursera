function submit(part)
  addpath('./lib');

  conf.assignmentKey = 'jk3STQNfEeadkApJXdJa6Q';
  conf.itemName = 'Simple BN Knowledge Engineering';

  conf.partArrays = { ...
    { ...
      'COq9M', ...
      { 'Credit_net.net' }, ...
      'Constructing a Credit Network', ...
    }, ...
    { ...
      '1AayO', ...
      { 'FactorProduct.m' }, ...
      'Factor Network', ...
    }, ...
    { ...
      'bsUeZ', ...
      { 'FactorProduct.m' }, ...
      'Factor Network (Test)', ...
    }, ...
    { ...
      'Y0uy4', ...
      { 'FactorMarginalization.m' }, ...
      'Factor Marginalization', ...
    }, ...
    { ...
      '5nfbi', ...
      { 'FactorMarginalization.m' }, ...
      'Factor Marginalization (Test)', ...
    }, ...
    { ...
      'VNVAv', ...
      { 'ObserveEvidence.m' }, ...
      'Observing Evidence', ...
    }, ...
    { ...
      'c8RCS', ...
      { 'ObserveEvidence.m' }, ...
      'Observing Evidence (Test)', ...
    }, ...
    { ...
      'rFh8a', ...
      { 'ComputeJointDistribution.m' }, ...
      'Computing the Joint Distribution', ...
    }, ...
    { ...
      'ThtGy', ...
      { 'ComputeJointDistribution.m' }, ...
      'Computing the Joint Distribution (Test)', ...
    }, ...
    { ...
      'lM96X', ...
      { 'ComputeMarginal.m' }, ...
      'Computing Marginals', ...
    }, ...
    { ...
      'gde9N', ...
      { 'ComputeMarginal.m' }, ...
      'Computing Marginals (Test)', ...
    }, ...
  };

  conf.output = @output;
  submitWithConfiguration(conf);

end

function out = output(partIdx)

  load submit_input;
  if partIdx == 1
    [F, names, assignments] = ConvertNetwork('Credit_net.net');
    [F, names, assignments] = RelabelVars(F, names, assignments);
    if (ValidateNetwork(F, names, assignments))      
      out = SerializeFactorsFg(F);
    else
      out = '';
    end
  elseif partIdx == 2
    fp = FactorProduct(PART2.SAMPLEINPUT{:});
    out = SerializeFactorsFg(fp);
  elseif partIdx == 3
    fp = [FactorProduct(PART2.INPUT1{:}), FactorProduct(PART2.INPUT2{:})];
    out = SerializeFactorsFg(fp);
  elseif partIdx == 4
    fm = [FactorMarginalization(PART3.SAMPLEINPUT{:})];
    out = SerializeFactorsFg(fm);
  elseif partIdx == 5
    fm = [FactorMarginalization(PART3.INPUT1{:}), FactorMarginalization(PART3.INPUT2{:})];  
    out = SerializeFactorsFg(fm);
  elseif partIdx == 6
    oe = [ObserveEvidence(PART4.SAMPLEINPUT{:})];
    out = SerializeFactorsFg(oe);
  elseif partIdx == 7
    oe = [ObserveEvidence(PART4.INPUT1{:})];
    out = SerializeFactorsFg(oe);
  elseif partIdx == 8
    jd = [ComputeJointDistribution(PART5.SAMPLEINPUT{:})];
    out = SerializeFactorsFg(jd);
  elseif partIdx == 9
    jd = [ComputeJointDistribution(PART5.INPUT1{:})];
    out = SerializeFactorsFg(jd);
  elseif partIdx == 10
    cm = [ComputeMarginal(PART6.SAMPLEINPUT{:})];
    out = SerializeFactorsFg(cm);
  elseif partIdx == 11
    cm = [ComputeMarginal(PART6.INPUT1{:}), ComputeMarginal(PART6.INPUT2{:}), ComputeMarginal(PART6.INPUT3{:}), ComputeMarginal(PART6.INPUT4{:})];
    out = SerializeFactorsFg(cm);
  end 
end

function ok = ValidateNetwork(F, names, assignments)
  % Check number of variables in network
  assert(numel(names) == 8, sprintf('Error: your network should have 8 variables but it only has %d. Did you delete some variables?', numel(names)));

  % Check number of CPDs in network
  % If this happens, it's most likely a malformed .net file, or else it is
  % a bug in ConvertNetwork.m
  assert(numel(F) == 8, sprintf('Error: your network should have 8 CPDs but it only has %d.', numel(F)));   
  
  % Check that the assignment of variables is the same as in the original
  valNames = cell(8, 1);
  valNames{1} = {'Low','High'};
  valNames{2} = {'High','Medium','Low'};
  valNames{3} = {'Positive','Negative'};
  valNames{4} = {'High','Medium','Low'};
  valNames{5} = {'Excellent','Acceptable','Unacceptable'};
  valNames{6} = {'Promising','Not_promising'};
  valNames{7} = {'Reliable','Unreliable'};
  valNames{8} = {'Between16and21','Between22and64','Over65'};
  
  for i = 1:numel(valNames)    
    assert(all(strcmp(valNames{i}, assignments{i})), ...
      sprintf('Error: variable values/value ordering for variable %s does not match the original. Did you change the order/name of some variable values?', names{i}));
  end
  
  ok = true;
end

% Relabel variable ids according to the grader ordering
function [newF, newNames, newAssignments] = RelabelVars(F, names, assignments)
  varNames = cell(8, 1);
  varNames{1} = 'DebtIncomeRatio';
  varNames{2} = 'Assets';
  varNames{3} = 'CreditWorthiness';
  varNames{4} = 'Income';
  varNames{5} = 'PaymentHistory';
  varNames{6} = 'FutureIncome';
  varNames{7} = 'Reliability';
  varNames{8} = 'Age';

  newToOrig = zeros(8, 1);
  origToNew = zeros(8, 1);
  for i = 1:8
    result = find(strcmp(varNames{i}, names), 1);
    if (isempty(result)) % no match!
      error('Variable %s not found in network! Did you change the variable names/identifiers?', varNames{i});
    else
      newToOrig(result) = i;
      origToNew(i) = result;
    end
  end
  
  newNames = names(origToNew(:));
  newAssignments = assignments(origToNew(:));
  
  % relabel the vars
  newF = F;
  for i = 1:numel(F)
    newF(i).var = newToOrig(F(i).var);
  end
end

function out = SerializeFactorsFg(F)
% Serializes a factor struct array into the .fg format for libDAI
% http://cs.ru.nl/~jorism/libDAI/doc/fileformats.html
%
% To avoid incompatibilities with EOL markers, make sure you write the
% string to a file using the appropriate file type ('wt' for windows, 'w'
% for unix)

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
