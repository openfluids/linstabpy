function export_resolvent_debug(L0_postBC, omega, nEig, W, invW, B, C, filters, export_dir, varargin)
%EXPORT_RESOLVENT_DEBUG Export full MATLAB resolvent intermediates for parity.
%
% This helper exports all intermediate matrices/actions needed to compare
% MATLAB and Python resolvent pipelines one-to-one.
%
% Inputs
%   L0_postBC : sparse complex operator after BCs
%   omega     : analysis angular frequency
%   nEig      : number of singular triplets to compute/export
%   W, invW   : energy metric and inverse (scalar/vector/sparse matrix)
%   B, C      : forcing and observation operators (scalar/vector/sparse matrix)
%   filters   : struct with fields filter and filter_ct (or [] for identity)
%   export_dir: output folder (e.g. fullfile(case_dir,'MATLAB_exports_resolvent'))
%
% Optional name-value pairs
%   'Seed'      : RNG seed for probes (default 20260226)
%   'NProbe'    : number of deterministic probe vectors (default 8)
%   'CaseName'  : metadata string (default "unspecified_case")
%   'SolverType': metadata string from GetInverseFunction branch (default "lu")
%   'Meta'      : struct with extra metadata fields (default struct())
%
% Example call from a case script (after assembling L0/W/B/C):
%   export_resolvent_debug(L0, omega, nEig, W, invW, B, C, mesh.filters, ...
%       fullfile('example_6_hamada','MATLAB_exports_resolvent'), ...
%       'CaseName','example_6_hamada');

% --------------------------- parse options -------------------------------
p = inputParser;
p.addParameter('Seed', 20260226, @(x) isnumeric(x) && isscalar(x));
p.addParameter('NProbe', 8, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('CaseName', "unspecified_case", @(x) ischar(x) || isstring(x));
p.addParameter('SolverType', "lu", @(x) ischar(x) || isstring(x));
p.addParameter('Meta', struct(), @isstruct);
p.parse(varargin{:});
opts = p.Results;

if ~issparse(L0_postBC)
    L0_postBC = sparse(L0_postBC);
end
nDOF = size(L0_postBC, 1);
if size(L0_postBC, 2) ~= nDOF
    error('L0_postBC must be square.');
end

if ~exist(export_dir, 'dir')
    mkdir(export_dir);
end

% ------------------------ normalize W/invW/B/C ---------------------------
if ~exist('W', 'var') || isempty(W)
    W = 1;
end
if ~exist('invW', 'var') || isempty(invW)
    invW = 1;
end
if ~exist('B', 'var') || isempty(B)
    B = 1;
end
if ~exist('C', 'var') || isempty(C)
    C = 1;
end

if isscalar(B)
    if B == 1
        B = speye(nDOF);
    else
        B = B * speye(nDOF);
    end
elseif size(B, 1) == 1 || size(B, 2) == 1
    B = spdiags(B(:), 0, nDOF, nDOF);
else
    B = sparse(B);
end

if isscalar(C)
    if C == 1
        C = speye(nDOF);
    else
        C = C * speye(nDOF);
    end
elseif size(C, 1) == 1 || size(C, 2) == 1
    C = spdiags(C(:), 0, nDOF, nDOF);
else
    C = sparse(C);
end

if isscalar(W)
    W_diag = full(W * ones(nDOF, 1));
    invW_diag = full((1 / W) * ones(nDOF, 1));
    W = spdiags(W_diag, 0, nDOF, nDOF);
    invW = spdiags(invW_diag, 0, nDOF, nDOF);
elseif size(W, 1) == 1 || size(W, 2) == 1
    W_diag = full(W(:));
    invW_diag = full(1 ./ W_diag);
    W = spdiags(W_diag, 0, nDOF, nDOF);
    invW = spdiags(invW_diag, 0, nDOF, nDOF);
else
    W = sparse(W);
    if isscalar(invW)
        error('invW scalar is invalid when W is matrix.');
    end
    if size(invW, 1) == 1 || size(invW, 2) == 1
        invW = spdiags(invW(:), 0, nDOF, nDOF);
    else
        invW = sparse(invW);
    end
    W_diag = full(diag(W));
    invW_diag = full(diag(invW));
end

if any(~isfinite(W_diag)) || any(~isfinite(invW_diag))
    error('W/invW diagonals contain non-finite values.');
end

% --------------------------- shifted operator ----------------------------
LsI = L0_postBC - 1i * omega * speye(nDOF);
[LL, UU, pp, qq, rr] = lu(LsI);

R = @(v) qq * (UU \ (LL \ (pp * (rr \ v))));
R_T = @(v) rr' \ (pp' * (LL' \ (UU' \ (qq' * v))));

% ----------------------------- filtering --------------------------------
if exist('filters', 'var') && ~isempty(filters) ...
        && isfield(filters, 'filter') && isfield(filters, 'filter_ct')
    FILTER = @(x) reshape(filters.filter(reshape(x, [], 5)), [], 1);
    FILTER_ct = @(x) reshape(filters.filter_ct(reshape(x, [], 5)), [], 1);
else
    FILTER = @(x) x;
    FILTER_ct = @(x) x;
end

H = @(v) FILTER(C * R(B * FILTER(v)));
Htr = @(v) FILTER_ct(B' * R_T(C' * FILTER_ct(v)));
HtrH = @(v) invW * Htr(W * H(v));

% ----------------------- deterministic probe export ----------------------
rng(opts.Seed, 'twister');
nProbe = double(opts.NProbe);
V_probe = randn(nDOF, nProbe) + 1i * randn(nDOF, nProbe);
FILTER_V_probe = zeros(nDOF, nProbe);
FILTERct_V_probe = zeros(nDOF, nProbe);
H_V_probe = zeros(nDOF, nProbe);
Htr_V_probe = zeros(nDOF, nProbe);
HtrH_V_probe = zeros(nDOF, nProbe);

for k = 1:nProbe
    vk = V_probe(:, k);
    FILTER_V_probe(:, k) = FILTER(vk);
    FILTERct_V_probe(:, k) = FILTER_ct(vk);
    H_V_probe(:, k) = H(vk);
    Htr_V_probe(:, k) = Htr(vk);
    HtrH_V_probe(:, k) = HtrH(vk);
end

% -------------------------- resolvent eig/SVD ---------------------------
eigs_opts.tol = eps;
eigs_opts.disp = 2;
eigs_opts.issym = false;
eigs_opts.isreal = false;

[V_raw, OMEGA_raw] = eigs(HtrH, nDOF, nEig, 'lm', eigs_opts);
S = sqrt(diag(real(OMEGA_raw)));
V = V_raw;
U = zeros(size(V_raw));

for i = 1:nEig
    den = sqrt(V(:, i)' * W * V(:, i));
    if abs(den) > 0
        V(:, i) = V(:, i) / den;
    end
    if abs(S(i)) > 0
        U(:, i) = H(V(:, i)) / S(i);
    else
        U(:, i) = H(V(:, i));
    end
end

gram_V = V' * W * V;
gram_U = U' * W * U;
residual_sigma = zeros(nEig, 1);
for i = 1:nEig
    lhs = H(V(:, i));
    rhs = S(i) * U(:, i);
    residual_sigma(i) = norm(lhs - rhs) / max(norm(rhs), 1e-30);
end

% ------------------------------- metadata --------------------------------
res_config = struct();
res_config.case_name = char(opts.CaseName);
res_config.generated_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
res_config.matlab_release = version('-release');
res_config.matlab_version = version;
res_config.nDOF = nDOF;
res_config.omega = omega;
res_config.nEig = nEig;
res_config.solver_type = char(opts.SolverType);
res_config.seed = opts.Seed;
res_config.nProbe = nProbe;
res_config.nnz = struct( ...
    'L0_postBC', nnz(L0_postBC), ...
    'LsI', nnz(LsI), ...
    'W', nnz(W), ...
    'invW', nnz(invW), ...
    'B', nnz(B), ...
    'C', nnz(C));
res_config.shape = struct( ...
    'L0_postBC', size(L0_postBC), ...
    'LsI', size(LsI), ...
    'W', size(W), ...
    'invW', size(invW), ...
    'B', size(B), ...
    'C', size(C));
res_config.extra_meta = opts.Meta;

% -------------------------------- saves ----------------------------------
save(fullfile(export_dir, 'res_config.mat'), 'res_config', '-v7.3');
save(fullfile(export_dir, 'L0_postBC.mat'), 'L0_postBC', '-v7.3');
save(fullfile(export_dir, 'W.mat'), 'W', '-v7.3');
save(fullfile(export_dir, 'invW.mat'), 'invW', '-v7.3');
save(fullfile(export_dir, 'W_diag.mat'), 'W_diag', '-v7.3');
save(fullfile(export_dir, 'invW_diag.mat'), 'invW_diag', '-v7.3');
save(fullfile(export_dir, 'B.mat'), 'B', '-v7.3');
save(fullfile(export_dir, 'C.mat'), 'C', '-v7.3');
save(fullfile(export_dir, 'LsI.mat'), 'LsI', '-v7.3');
save(fullfile(export_dir, 'LU_factors.mat'), 'LL', 'UU', 'pp', 'qq', 'rr', '-v7.3');
save(fullfile(export_dir, 'res_probe_vectors.mat'), 'V_probe', '-v7.3');
save(fullfile(export_dir, 'res_probe_filter.mat'), 'FILTER_V_probe', 'FILTERct_V_probe', '-v7.3');
save(fullfile(export_dir, 'res_probe_actions.mat'), 'H_V_probe', 'Htr_V_probe', 'HtrH_V_probe', '-v7.3');
save(fullfile(export_dir, 'res_svd_raw.mat'), 'OMEGA_raw', 'V_raw', '-v7.3');
save(fullfile(export_dir, 'res_svd_final.mat'), 'S', 'U', 'V', '-v7.3');
save(fullfile(export_dir, 'res_checks.mat'), 'gram_V', 'gram_U', 'residual_sigma', '-v7.3');

fprintf('[export_resolvent_debug] wrote artifacts to: %s\n', export_dir);
end
