function s = martini2024_pack_parity_export(kind, varargin)
%MARTINI2024_PACK_PARITY_EXPORT Trim MATLAB bundles to parity/plot needs.
%
% Keeps only the fields consumed by extract_martini_to_npz.py,
% compare_martini_examples.py, and example_*_plots.py.
%
% Resolvent exports store compact u+v mode columns (see
% martini2024_compact_resolvent_uv) so Python split_martini_fixtures.py and
% plotters do not depend on discovering a particular struct prefix.

switch kind
    case 'resolvent'
        gains = varargin{1};
        responses = varargin{2};
        forces = varargin{3};
        omega = varargin{4};
        nEig = varargin{5};
        mesh = varargin{6};
        idx = varargin{7};
        nKeep = varargin{8};
        nk = min(nKeep, numel(gains));
        forces_k = forces(:, 1:nk);
        responses_k = responses(:, 1:nk);
        [forces_k, responses_k, idx_uv] = martini2024_compact_resolvent_uv( ...
            forces_k, responses_k, idx);
        s = struct( ...
            'gains', gains(1:nk), ...
            'responses', responses_k, ...
            'forces', forces_k, ...
            'omega', omega, ...
            'nEig', nEig, ...
            'plot_grid', martini2024_pack_plot_grid(mesh), ...
            'idx', idx_uv, ...
            'mode_plot_indices', 1:nk);

    case 'resolvent_jet'
        gains = varargin{1};
        responses = varargin{2};
        forces = varargin{3};
        omega = varargin{4};
        st = varargin{5};
        nEig = varargin{6};
        mesh = varargin{7};
        idx = varargin{8};
        nKeep = varargin{9};
        s = martini2024_pack_parity_export( ...
            'resolvent', gains, responses, forces, omega, nEig, mesh, idx, nKeep);
        s.St = st;

    case 'temporal'
        omega = varargin{1};
        V = varargin{2};
        iplot = varargin{3};
        meta = varargin{4};
        mesh = varargin{5};
        idx = varargin{6};
        nKeep = varargin{7};
        nk = min(nKeep, size(V, 2));
        s = struct( ...
            'omega', omega, ...
            'V', V(:, 1:nk), ...
            'iplot', iplot(1:min(numel(iplot), nk)), ...
            'plot_grid', martini2024_pack_plot_grid(mesh), ...
            'idx', martini2024_pack_idx(idx));
        meta_fields = {'alpha_tempAn', 'Lx', 'Ly', 'nPeriods', 'floquetExp', ...
            'Nx', 'Ny', 'time_eigsMatlab'};
        for k = 1:numel(meta_fields)
            fn = meta_fields{k};
            if isfield(meta, fn)
                s.(fn) = meta.(fn);
            end
        end

    case 'spatial'
        alphas = varargin{1};
        alpha_target = varargin{2};
        omega_driver = varargin{3};
        iplot = varargin{4};
        wi = varargin{5};
        meta = varargin{6};
        mesh = varargin{7};
        idx = varargin{8};
        s = struct( ...
            'alphas', alphas, ...
            'alpha_target', alpha_target, ...
            'omega_driver', omega_driver, ...
            'iplot', iplot, ...
            'wi', wi, ...
            'plot_grid', martini2024_pack_plot_grid(mesh), ...
            'idx', martini2024_pack_idx(idx));
        meta_fields = {'Lx', 'nPeriods', 'floquetExp'};
        for k = 1:numel(meta_fields)
            fn = meta_fields{k};
            if isfield(meta, fn)
                s.(fn) = meta.(fn);
            end
        end

    case 'cavity_direct'
        omega = varargin{1};
        V = varargin{2};
        iplot = varargin{3};
        omega_target = varargin{4};
        mesh = varargin{5};
        idx = varargin{6};
        nKeep = varargin{7};
        nk = min(nKeep, size(V, 2));
        s = struct( ...
            'omega', omega, ...
            'V', V(:, 1:nk), ...
            'iplot', iplot(1:min(numel(iplot), nk)), ...
            'omega_target', omega_target, ...
            'plot_grid', martini2024_pack_plot_grid(mesh), ...
            'idx', martini2024_pack_idx(idx));

    case 'cavity_adjoint'
        omega_adj = varargin{1};
        V_adj = varargin{2};
        iplot_adj = varargin{3};
        nKeep = varargin{4};
        nk = min(nKeep, size(V_adj, 2));
        s = struct( ...
            'omegaAdj', omega_adj, ...
            'VAdj', V_adj(:, 1:nk), ...
            'iplot_adj', iplot_adj(1:min(numel(iplot_adj), nk)));

    otherwise
        error('martini2024_pack_parity_export: unknown kind ''%s''.', kind);
end
end
