function submitWithConfiguration(conf)
  addpath('./lib/jsonlab');

  parts = parts(conf);

  fprintf('== Submitting solutions | %s...\n', conf.itemName);

  tokenFile = 'token.mat';
  if exist(tokenFile, 'file')
    load(tokenFile);
    [email token] = promptToken(email, token, tokenFile);
  else
    [email token] = promptToken('', '', tokenFile);
  end

  if isempty(token)
    fprintf('!! Submission Cancelled\n');
    return
  end

  try
    response = submitParts(conf, email, token, parts);
  catch
    e = lasterror();
    fprintf( ...
      '!! Submission failed: unexpected error: %s\n', ...
      e.message);
    fprintf('!! Please try again later.\n');
    return
  end

  if isfield(response, 'errorMessage')
    fprintf('!! Submission failed: %s\n', response.errorMessage);
  else
    fprintf('Submission successful. You can view your grade under My Submission on the programming assignment page.\n\n');
    save(tokenFile, 'email', 'token');
  end
end

function [email token] = promptToken(email, existingToken, tokenFile)
  if (~isempty(email) && ~isempty(existingToken))
    prompt = sprintf( ...
      'Use token from last successful submission (%s)? (Y/n): ', ...
      email);
    reenter = input(prompt, 's');

    if (isempty(reenter) || reenter(1) == 'Y' || reenter(1) == 'y')
      token = existingToken;
      return;
    else
      delete(tokenFile);
    end
  end
  email = input('Login (email address): ', 's');
  token = input('Token: ', 's');
end

function isValid = isValidPartOptionIndex(partOptions, i)
  isValid = (~isempty(i)) && (1 <= i) && (i <= numel(partOptions));
end

function response = submitParts(conf, email, token, parts)
  body = makePostBody(conf, email, token, parts);
  submissionUrl = submissionUrl();
  params = {'jsonBody', body};
  responseBody = urlread(submissionUrl, 'post', params);
  response = loadjson(responseBody);
end

function body = makePostBody(conf, email, token, parts)
  bodyStruct.assignmentKey = conf.assignmentKey;
  bodyStruct.submitterEmail = email;
  bodyStruct.secret = token;
  bodyStruct.parts = makePartsStruct(conf, parts);

  opt.Compact = 1;
  body = savejson('', bodyStruct, opt);
end

function partsStruct = makePartsStruct(conf, parts)
  partIdx = 0;
  for part = parts
    partId = part{:}.id;
    partIdx = partIdx + 1;
    fieldName = makeValidFieldName(partId);
    outputStruct.output = conf.output(partIdx);
    partsStruct.(fieldName) = outputStruct;
  end
end

function [parts] = parts(conf)
  parts = {};
  for partArray = conf.partArrays
    part.id = partArray{:}{1};
    part.sourceFiles = partArray{:}{2};
    part.name = partArray{:}{3};
    parts{end + 1} = part;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Service configuration
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function submissionUrl = submissionUrl()
  submissionUrl = 'https://www.coursera.org/api/onDemandProgrammingScriptSubmissionsController.v1';
end
