clear all
close all
clc

% % check center (not quite useful)
dim = 384;
% center_cpp_fp = fopen('GIST80M-CENTER-float.dat', 'rb');
% fread(center_cpp_fp, 1, 'int');
% center_cpp = fread(center_cpp_fp, [dim, 1], 'single');
% center_mat = load('center.mat');
% center_mat = center_mat.learn_mean;
% 
% diff = norm(center_cpp - center_mat, 2);
% 
% fprintf('%.8f\n', diff);


% % check codes
n = 1e6;
m = 384;
enc_dim = m / 64;
codes_cpp_fp = fopen('gist80m_codes.bin', 'rb');
codes_cpp = uint64(fread(codes_cpp_fp, [enc_dim, n], 'uint64'));
fclose(codes_cpp_fp);
codes_mat = load('compact_gist.mat');
codes_mat = codes_mat.B1;
codes_diffs = codes_mat - codes_cpp;
final_diff = sum(abs(codes_diffs(:)));