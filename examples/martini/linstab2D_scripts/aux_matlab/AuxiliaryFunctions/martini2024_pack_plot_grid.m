function s = martini2024_pack_plot_grid(mesh)
%MARTINI2024_PACK_PLOT_GRID Minimal grid for mode plots (not full mesh operators).
s = struct();
s.X = mesh.X;
s.Y = mesh.Y;
if isfield(mesh, 'usedInd')
    s.usedInd = mesh.usedInd;
end
end
