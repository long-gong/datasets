% % This MATLAB script was created ONLY for debug purpose !!
% % Some codes are copied from the source codes provided by MIH (Multi-Indexing Hashing) paper
% % https://www.cs.toronto.edu/~norouzi/research/papers/multi_index_hashing.pdf
ds_dir = '/home/saber/Dropbox/CodeHub/bvecs-extract';
ds_filename = [ds_dir '/tinygist80million.bin'];
hash_filename = 'gaussian_mat.txt';
dim = 384;
m = 384;
n = 1e6;
word_size = 64;
fp = fopen(ds_filename, 'rb');
A = fread(fp, [dim, n], 'single');
A = double(A);

center_cpp_fp = fopen('GIST80M-CENTER-float.dat', 'rb');
fread(center_cpp_fp, 1, 'int');
center_cpp = fread(center_cpp_fp, [dim, 1], 'single');
A = bsxfun(@minus,A, double(center_cpp));

% % load hash function to make sure the same hash functions are used
W = [load(hash_filename) zeros(m, 1)]; 
raw = (W * [A; ones(1, size(A,2))]);
B1 =  raw >= 0;
B1 = compactbit(B1, word_size);
save ('compact_gist.mat', 'B1');
codes_bin = fopen('gist80m_codes_matlab.bin', 'wb');
fwrite(codes_bin, B1, 'uint64');
fclose(codes_bin);
% % remove deplications
B2 = unique(B1', 'rows');
save ('unique_compact_gist.mat', 'B2')