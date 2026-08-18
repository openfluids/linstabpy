function c = martini2024_parity_constants()
%MARTINI2024_PARITY_CONSTANTS Mode counts aligned with martini_cases.yaml n_compare.
%
% Plotters use N_MODES=3; parity gates need at least n_compare modes saved.
% Saving max(plot, parity) keeps one regeneration path for both workflows.

c.N_RESOLVENT = 3;       % Examples 1, 2, 5 — compare n_compare=3
c.N_CAVITY = 6;          % Example 4 direct/adjoint — compare n_compare=6
c.N_EX3_TEMPORAL = 50;   % Example 3 temporal — full Arnoldi export (nEig in driver)
end
