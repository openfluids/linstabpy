function save_martini2024_bundle(results_stem, logs_stem, results_struct, logs_struct)
%SAVE_MARTINI2024_BUNDLE Write results/logs .mat (v7.3) into the flat output root.
outdir = martini2024_output_root();
results_file = fullfile(outdir, [results_stem, '.mat']);
logs_file = fullfile(outdir, [logs_stem, '.mat']);

results = results_struct;
save(results_file, 'results', '-v7.3');
fprintf('Saved results: %s\n', results_file);

if nargin >= 4 && ~isempty(logs_struct)
    logs = logs_struct;
    save(logs_file, 'logs', '-v7.3');
    fprintf('Saved logs: %s\n', logs_file);
end
end
