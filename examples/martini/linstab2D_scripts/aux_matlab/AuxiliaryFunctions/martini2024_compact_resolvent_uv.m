function [forces_c, responses_c, idx_c] = martini2024_compact_resolvent_uv(forces, responses, idx)
%MARTINI2024_COMPACT_RESOLVENT_UV Keep u+v DOFs only for resolvent parity NPZ.
%
% Examples 1–2 force/observe rho,w,T through B=C masking but presentation
% fixtures store compact u-then-v mode columns. Global idx.u_j / idx.v_j are
% preserved so Python split_martini_fixtures.py and plotters can map DOFs
% consistently (plot code renumbers to compact columns only for gridding).

uj = idx.u_j(:);
vj = idx.v_j(:);
forces_c = [forces(uj, :); forces(vj, :)];
responses_c = [responses(uj, :); responses(vj, :)];

idx_c = struct();
idx_c.u_j = uj;
idx_c.v_j = vj;
end
