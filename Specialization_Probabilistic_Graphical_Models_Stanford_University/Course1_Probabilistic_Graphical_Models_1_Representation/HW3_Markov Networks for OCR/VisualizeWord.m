function VisualizeWord (word)
% This function allows you to visualize the characters for a single word.
%
% Input:
%  word: A struct array, each with an 'img' attribute that gives the 16x8
%    pixel matrix for that image.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

padding = zeros(size(word(1).img, 1), 1);

totalWidth = 10 * length(word);
im = zeros(16, totalWidth);
for i = 1:length(word)
    charIm = [padding word(i).img padding];
    im(:, (1 + 10 * (i-1)) + (1:10)) = charIm;
end

width = size(im, 2);
padding = zeros(1, width);
im = [padding; im; padding];

figure;
colormap(gray);
imagesc(1 - im);
axis equal;
[height, width] = size(im);
axis([0 width 0 height]);

end

