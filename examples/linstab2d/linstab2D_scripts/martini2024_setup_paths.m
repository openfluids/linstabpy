function martini2024_setup_paths()
%MARTINI2024_SETUP_PATHS Add LinStab2D + local export helpers on the MATLAB path.
%
% Requires:
%   LINSTAB2D_ROOT  — upstream LinStab2D checkout (aux_matlab + readbaseflow .m)
%
% Creates BaseFlows -> ../baseflows symlink in this folder so load('BaseFlows/...')
% resolves to examples/linstab2d/baseflows/*.mat.

scripts_root = fileparts(mfilename('fullpath'));
addpath(scripts_root);
addpath(genpath(fullfile(scripts_root, 'aux_matlab')));

linstab2d = getenv('LINSTAB2D_ROOT');
if isempty(linstab2d)
    error(['martini2024: set LINSTAB2D_ROOT to a LinStab2D checkout ', ...
        '(a tree with aux_matlab/ and BaseFlows/).']);
end
if ~exist(linstab2d, 'dir')
    error('martini2024: LINSTAB2D_ROOT does not exist: %s', linstab2d);
end

addpath(genpath(fullfile(linstab2d, 'aux_matlab')));
addpath(fullfile(linstab2d, 'BaseFlows'));

bf_link = fullfile(scripts_root, 'BaseFlows');
bf_data = fullfile(scripts_root, '..', 'baseflows');
if ~exist(bf_data, 'dir')
    error('martini2024: missing examples/linstab2d/baseflows at %s', bf_data);
end
if ~exist(bf_link, 'dir')
    if ispc
        cmd = sprintf('mklink /J "%s" "%s"', bf_link, bf_data);
    else
        cmd = sprintf('ln -sfn "%s" "%s"', bf_data, bf_link);
    end
    status = system(cmd);
    if status ~= 0
        error('martini2024: could not link BaseFlows to baseflows (status=%d).', status);
    end
end
end
