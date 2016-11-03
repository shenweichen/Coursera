function show_rbm(rbm_w)
    n_hid = size(rbm_w, 1);
    n_rows = ceil(sqrt(n_hid));
    blank_lines = 4;
    distance = 16 + blank_lines;
    to_show = zeros([n_rows * distance + blank_lines, n_rows * distance + blank_lines]);
    for i = 0:n_hid-1,
        row_i = floor(i / n_rows);
        col_i = mod(i, n_rows);
        pixels = reshape(rbm_w(i+1, :), [16, 16]).';
        row_base = row_i*distance + blank_lines;
        col_base = col_i*distance + blank_lines;
        to_show(row_base+1:row_base+16, col_base+1:col_base+16) = pixels;
    end
    extreme = max(abs(to_show(:)));
    try
        imshow(to_show, [-extreme, extreme]);
        title('hidden units of the RBM');
    catch err
        fprintf('Failed to display the RBM. No big deal (you do not need the display to finish the assignment), but you are missing out on an interesting picture.\n');
    end
end

