function s = martini2024_pack_idx(idx)
%MARTINI2024_PACK_IDX Export DOF index ranges needed to reshape modes.
s = struct();
names = {'u_i', 'u_j', 'v_i', 'v_j', 'w_i', 'w_j', 'T_i', 'T_j', 'rho_i', 'rho_j'};
for k = 1:numel(names)
    nm = names{k};
    if isfield(idx, nm)
        s.(nm) = idx.(nm);
    end
end
end
