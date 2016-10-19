function pred = RunInference (factors)
% This function performs inference for a Markov network specified as a list
% of factors.
%
% Input:
%   factors: An array of struct factors, each containing 'var', 'card', and
%     'val' fields.
%
% Output:
%   pred: An array of predictions for every variable. In particular,
%     pred(i) is the predicted value for variable numbered i (as determined
%     by the 'var' fields in the input factors).
%
% Copyright (C) Daphne Koller, Stanford University, 2012

binaries = {'.\inference\doinference.exe', ...
            './inference/doinference-mac', ...
            './inference/doinference-linux'};

kFactorsFilename = 'factors.fg';
kStderrFilename = 'inf.log';
kInfBinary = binaries{[ispc ismac isunix]}; % NB: need ismac first so that if ismac and isunix are both 1, then mac is chosen
kInferenceType = 'map'; % choices are 'map' or 'pd'

factorsString = SerializeFactorsFg (factors);

fd = fopen(kFactorsFilename, 'wt');
fprintf (fd, '%s', factorsString);
fclose(fd);

if (isunix && ~ismac) 
  command = [kInfBinary ' ' kFactorsFilename ' ' kInferenceType];
else
  command = [kInfBinary ' ' kFactorsFilename ' ' kInferenceType ' 2> ' kStderrFilename];
end

[retVal, output] = system(command);

if (retVal ~= 0)
    error('The doinference command failed. Look at the file %s to diagnose the cause', kStderrFilename);
end

pred = ParseOutput(output);

end

function pred = ParseOutput(output)

lines = strread(output, '%s', 'delimiter', sprintf('\n'));
lines(strcmp(lines, '')) = [];

numVars = str2double(lines{1});

if (numVars ~= length(lines) - 1)
    error('Error parsing output: %s', output);
end

pred = zeros(numVars, 1);
for i = 2:(numVars + 1)
    line = str2num(lines{i}); %#ok
    pred(i-1) = line(end);
end

end
