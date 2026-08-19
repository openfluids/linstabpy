function root = martini2024_output_root()
%MARTINI2024_OUTPUT_ROOT Flat directory for parity .mat exports (outside Git).
root = getenv('MARTINI2024_OUTPUT_ROOT');
if isempty(root)
    helper_dir = fileparts(mfilename('fullpath'));
    repo_root = helper_dir;
    for k = 1:5
        repo_root = fileparts(repo_root);
    end
    root = fullfile(repo_root, 'martini_examples');
end
root = char(root);
martini2024_mkdir_p(root);
end

function martini2024_mkdir_p(target_dir)
if exist(target_dir, 'dir')
    return;
end
parent = fileparts(target_dir);
if ~isempty(parent) && ~exist(parent, 'dir')
    martini2024_mkdir_p(parent);
end
mkdir(target_dir);
end
