function [L0,idx] = GetLinProblem_export(mesh,BF,model,mkx,floquetExp)
    % GetLinProblem_export
    % Wrapper around the original GetLinProblem that exports internal
    % matrices and coefficients used to build L0, for Python parity checks.
    %
    % Usage in main_export_stages.m:
    %   [L0, idx] = GetLinProblem_export(mesh, baseFlow, '2D', beta);
    %
    % This writes into MATLAB_exports_Ziming_Internal/:
    %   - derivatives.mat      (Dr, D2r, Dz, D2z, DR, D2R, DZ, D2Z, D2RZ)
    %   - coefficients_raw.mat (A0_ij, Ar_ij, Az_ij, Arr_ij, Azz_ij, Arz_ij, B_ij)
    %   - lhs_rhs_blocks.mat   (A0, Ar, Az, Arr, Azz, Arz, RHS, LHS)
    %   - L0_internal.mat      (L0_before_T, L0, idx)

    if ~exist('floquetExp','var'); floquetExp = 0; end
    if ~exist('model','var');      model      = '2D'; end

    % ---------------------------------------------------------------------
    % Export directory
    % ---------------------------------------------------------------------
    this_dir   = fileparts(mfilename('fullpath'));
    export_dir = fullfile(this_dir, 'MATLAB_exports_Ziming_Internal');
    if exist(export_dir, 'dir') ~= 7
        mkdir(export_dir);
    end

    fprintf('--- Creating linear operators for model %s and floquet exp=%3.1f...', model, floquetExp); tic();

    % ---------------------------------------------------------------------
    % Mesh derivatives and coordinates
    % ---------------------------------------------------------------------
    z  = mesh.X(mesh.usedInd);
    Dz = mesh.Dx;
    D2z = mesh.D2x;

    r  = mesh.Y(mesh.usedInd);
    Dr = mesh.Dy;
    D2r = mesh.D2y;

    Drz = mesh.Dxy;
    Dzr = mesh.Dyx;

    nGridPoints = mesh.ngp;

    % ---------------------------------------------------------------------
    % Mean flow in CS1 (match original GetLinProblem.m)
    % ---------------------------------------------------------------------
    if strcmp(model,'2D')
        U   = -BF.W(mesh.usedInd);
        V   =  BF.V(mesh.usedInd);
        W   =  BF.U(mesh.usedInd);
        mkx = -mkx;
    else
        U   =  BF.V(mesh.usedInd);
        V   =  BF.W(mesh.usedInd);
        W   =  BF.U(mesh.usedInd);
    end

    RHO =  BF.RHO(mesh.usedInd);
    T   =  BF.T(mesh.usedInd);
    MU  =  BF.MU(mesh.usedInd);
    P   =  RHO.*T./(BF.Ma^2*BF.kappa);

    if isfield(BF,'dmudT')  ; dmudT     = BF.dmudT(mesh.usedInd);
    else                    ; dmudT     = zeros(size(MU)); end

    if isfield(BF,'d2mudT2'); d2mudT2   = BF.d2mudT2(mesh.usedInd);
    else                    ; d2mudT2   = zeros(size(MU)); end

    if isfield(BF,'dMUdT')  ; dMUdT     = BF.dMUdT(mesh.usedInd);
    else                    ; dMUdT     = zeros(size(MU)); end

    if isfield(BF,'d2MUdT2'); d2MUdT2   = BF.d2MUdT2(mesh.usedInd);
    else                    ; d2MUdT2   = zeros(size(MU)); end

    cv    = BF.cv;
    c1    = BF.c1;
    c2    = BF.c2;
    kappa = BF.kappa;

    % ---------------------------------------------------------------------
    % Baseflow derivatives
    % ---------------------------------------------------------------------
    dUdr    = Dr*U;     d2Udr2    = D2r*U;
    dVdr    = Dr*V;     d2Vdr2    = D2r*V;
    dWdr    = Dr*W;     d2Wdr2    = D2r*W;
    dRHOdr  = Dr*RHO;   d2RHOdr2  = D2r*RHO;
    dTdr    = Dr*T;     d2Tdr2    = D2r*T;
    dUdz    = Dz*U;     d2Udz2    = D2z*U;
    dVdz    = Dz*V;     d2Vdz2    = D2z*V;
    dWdz    = Dz*W;     d2Wdz2    = D2z*W;
    dRHOdz  = Dz*RHO;   d2RHOdz2  = D2z*RHO;
    dTdz    = Dz*T;     d2Tdz2    = D2z*T;
    d2Udrz  = Dz*dUdr;
    d2Vdrz  = Dz*dVdr;
    d2Wdrz  = Dz*dWdr;
    dPdr    = Dr*P;
    dPdz    = Dz*P;

    % ---------------------------------------------------------------------
    % Global diff. matrices and coefficients (DR, DZ, etc.)
    % ---------------------------------------------------------------------
    if strcmp(model,'2D')
        Ma  = BF.Ma;
        Pr  = BF.Pr;
        kz  = mkx;
        Re  = BF.Re;
        NrNz = nGridPoints;
        DR  = blkdiag(Dr ,Dr ,Dr ,Dr ,Dr );
        D2R = blkdiag(D2r,D2r,D2r,D2r,D2r);
        DZ  = blkdiag(Dz ,Dz ,Dz ,Dz ,Dz );
        D2Z = blkdiag(D2z,D2z,D2z,D2z,D2z);
        D2RZ= blkdiag(Dzr,Dzr,Dzr,Dzr,Dzr);
    else
        m   = mkx;
        Re  = BF.Re;
        [DR,D2R,DZ,D2Z,D2RZ,D2ZR] = CreateDiffMatrices_Axy(mesh,m);
    end

    if floquetExp ~= 0
        II = speye(size(DZ));
        DZ   = DZ   + 1i*floquetExp*II;
        D2RZ = D2RZ + 1i*floquetExp*II;
        D2Z  = D2Z  + 2i*floquetExp*DZ + (1i*floquetExp)^2*II;
    end

    % Export derivative matrices
    save(fullfile(export_dir, 'derivatives.mat'), ...
         'Dr','D2r','Dz','D2z','DR','D2R','DZ','D2Z','D2RZ', ...
         'nGridPoints','-v7');

    % ---------------------------------------------------------------------
    % Coefficient vectors via getCoeffsLaminar_Cartesian / getCoeffsLaminar
    % ---------------------------------------------------------------------
    if strcmp(model,'2D')
        getCoeffsLaminar_Cartesian
    else
        getCoeffsLaminar
    end

    % Save all A*_ij and B_ij coefficient arrays.
    save(fullfile(export_dir, 'coefficients_raw.mat'), ...
         '-regexp', ...
         '^A0_','^Ar_','^Az_','^Arr_','^Azz_','^Arz_','^B_', ...
         '-v7');

    % ---------------------------------------------------------------------
    % Build block matrices A0, Ar, Az, Arr, Azz, Arz, RHS (as in original)
    % ---------------------------------------------------------------------
    A0 =    ...
        [   ...
        diag(sparse(A0_11)) diag(sparse(A0_12)) diag(sparse(A0_13)) diag(sparse(A0_14)) diag(sparse(A0_15))
        diag(sparse(A0_21)) diag(sparse(A0_22)) diag(sparse(A0_23)) diag(sparse(A0_24)) diag(sparse(A0_25))
        diag(sparse(A0_31)) diag(sparse(A0_32)) diag(sparse(A0_33)) diag(sparse(A0_34)) diag(sparse(A0_35))
        diag(sparse(A0_41)) diag(sparse(A0_42)) diag(sparse(A0_43)) diag(sparse(A0_44)) diag(sparse(A0_45))
        diag(sparse(A0_51)) diag(sparse(A0_52)) diag(sparse(A0_53)) diag(sparse(A0_54)) diag(sparse(A0_55))
        ];

    Ar =    ...
        [   ...
        diag(sparse(Ar_11)) diag(sparse(Ar_12)) diag(sparse(Ar_13)) diag(sparse(Ar_14)) diag(sparse(Ar_15))
        diag(sparse(Ar_21)) diag(sparse(Ar_22)) diag(sparse(Ar_23)) diag(sparse(Ar_24)) diag(sparse(Ar_25))
        diag(sparse(Ar_31)) diag(sparse(Ar_32)) diag(sparse(Ar_33)) diag(sparse(Ar_34)) diag(sparse(Ar_35))
        diag(sparse(Ar_41)) diag(sparse(Ar_42)) diag(sparse(Ar_43)) diag(sparse(Ar_44)) diag(sparse(Ar_45))
        diag(sparse(Ar_51)) diag(sparse(Ar_52)) diag(sparse(Ar_53)) diag(sparse(Ar_54)) diag(sparse(Ar_55))
        ];

    Az =    ...
        [   ...
        diag(sparse(Az_11)) diag(sparse(Az_12)) diag(sparse(Az_13)) diag(sparse(Az_14)) diag(sparse(Az_15))
        diag(sparse(Az_21)) diag(sparse(Az_22)) diag(sparse(Az_23)) diag(sparse(Az_24)) diag(sparse(Az_25))
        diag(sparse(Az_31)) diag(sparse(Az_32)) diag(sparse(Az_33)) diag(sparse(Az_34)) diag(sparse(Az_35))
        diag(sparse(Az_41)) diag(sparse(Az_42)) diag(sparse(Az_43)) diag(sparse(Az_44)) diag(sparse(Az_45))
        diag(sparse(Az_51)) diag(sparse(Az_52)) diag(sparse(Az_53)) diag(sparse(Az_54)) diag(sparse(Az_55))
        ];

    Arz =   ...
        [   ...
        diag(sparse(Arz_11)) diag(sparse(Arz_12)) diag(sparse(Arz_13)) diag(sparse(Arz_14)) diag(sparse(Arz_15))
        diag(sparse(Arz_21)) diag(sparse(Arz_22)) diag(sparse(Arz_23)) diag(sparse(Arz_24)) diag(sparse(Arz_25))
        diag(sparse(Arz_31)) diag(sparse(Arz_32)) diag(sparse(Arz_33)) diag(sparse(Arz_34)) diag(sparse(Arz_35))
        diag(sparse(Arz_41)) diag(sparse(Arz_42)) diag(sparse(Arz_43)) diag(sparse(Arz_44)) diag(sparse(Arz_45))
        diag(sparse(Arz_51)) diag(sparse(Arz_52)) diag(sparse(Arz_53)) diag(sparse(Arz_54)) diag(sparse(Arz_55))
        ];

    Arr =   ...
        [   ...
        diag(sparse(Arr_11)) diag(sparse(Arr_12)) diag(sparse(Arr_13)) diag(sparse(Arr_14)) diag(sparse(Arr_15))
        diag(sparse(Arr_21)) diag(sparse(Arr_22)) diag(sparse(Arr_23)) diag(sparse(Arr_24)) diag(sparse(Arr_25))
        diag(sparse(Arr_31)) diag(sparse(Arr_32)) diag(sparse(Arr_33)) diag(sparse(Arr_34)) diag(sparse(Arr_35))
        diag(sparse(Arr_41)) diag(sparse(Arr_42)) diag(sparse(Arr_43)) diag(sparse(Arr_44)) diag(sparse(Arr_45))
        diag(sparse(Arr_51)) diag(sparse(Arr_52)) diag(sparse(Arr_53)) diag(sparse(Arr_54)) diag(sparse(Arr_55))
        ];

    Azz =   ...
        [   ...
        diag(sparse(Azz_11)) diag(sparse(Azz_12)) diag(sparse(Azz_13)) diag(sparse(Azz_14)) diag(sparse(Azz_15))
        diag(sparse(Azz_21)) diag(sparse(Azz_22)) diag(sparse(Azz_23)) diag(sparse(Azz_24)) diag(sparse(Azz_25))
        diag(sparse(Azz_31)) diag(sparse(Azz_32)) diag(sparse(Azz_33)) diag(sparse(Azz_34)) diag(sparse(Azz_35))
        diag(sparse(Azz_41)) diag(sparse(Azz_42)) diag(sparse(Azz_43)) diag(sparse(Azz_44)) diag(sparse(Azz_45))
        diag(sparse(Azz_51)) diag(sparse(Azz_52)) diag(sparse(Azz_53)) diag(sparse(Azz_54)) diag(sparse(Azz_55))
        ];

    ZZ = 0*speye(nGridPoints,nGridPoints);

    RHS =     ...
        [   ...
        diag(sparse(B_11)) diag(sparse(B_12)) diag(sparse(B_13)) diag(sparse(B_14)) diag(sparse(B_15))
        diag(sparse(B_21)) diag(sparse(B_22)) diag(sparse(B_23)) diag(sparse(B_24)) diag(sparse(B_25))
        diag(sparse(B_31)) diag(sparse(B_32)) diag(sparse(B_33)) diag(sparse(B_34)) diag(sparse(B_35))
        diag(sparse(B_41)) diag(sparse(B_42)) diag(sparse(B_43)) diag(sparse(B_44)) diag(sparse(B_45))
        diag(sparse(B_51)) diag(sparse(B_52)) diag(sparse(B_53)) diag(sparse(B_54)) diag(sparse(B_55))
        ];

    % ---------------------------------------------------------------------
    % LHS and export pre-B-inversion
    % ---------------------------------------------------------------------
    LHS = A0 + Ar*DR + Az*DZ + Arr*D2R + Azz*D2Z + Arz*D2RZ;
    if strcmp(model,'2D'); LHS = -LHS; end

    save(fullfile(export_dir, 'lhs_rhs_blocks.mat'), ...
         'A0','Ar','Az','Arr','Azz','Arz','RHS','LHS', ...
         '-v7');

    % ---------------------------------------------------------------------
    % Build idx (same as original GetLinProblem.m)
    % ---------------------------------------------------------------------
    idx = struct();
    ngp = mesh.ngp;
    idx = mesh.idx;

    boundaries = fields(mesh.idx);
    variables  = {'rho','u','v','w','T'};
    for ib = 1:length(boundaries)
        b  = boundaries{ib};
        idx.([b '_all']) = [];
        for iv = 1:length(variables)
            v = variables{iv};
            idx.([b '_' v]) =  idx.(b) + ngp*(iv-1);
            idx.([b '_all'])=  [idx.([b '_all']); idx.([b '_' v])] ;
        end
    end
    for iv = 1:length(variables)
        v = variables{iv};
        idx.([v '_j']) =  (1:ngp) + ngp*(iv-1);
    end

    % ---------------------------------------------------------------------
    % Solve for L0, apply sponge, and T transform
    % ---------------------------------------------------------------------
    L0_before_T = (RHS/1i)\LHS;

    if isfield(mesh,'sponge')
        sponge  = mesh.sponge(mesh.usedInd);
        Asponge = spdiags( repmat(sponge(:),5,1),0,ngp*5,ngp*5);
        L0_before_T = L0_before_T - Asponge;
    end

    II = speye(size(Dz));
    ZZ = II*0;

    if strcmp(model,'2D')
        T  = [  II, ZZ, ZZ, ZZ, ZZ;
                ZZ, ZZ, ZZ, II, ZZ;
                ZZ, ZZ, II, ZZ, ZZ;
                ZZ,-II, ZZ, ZZ, ZZ;
                ZZ, ZZ, ZZ, ZZ, II ];
    else
        T  = [  II, ZZ, ZZ, ZZ, ZZ;
                ZZ, ZZ, ZZ, II, ZZ;
                ZZ, II, ZZ, ZZ, ZZ;
                ZZ, ZZ, II, ZZ, ZZ;
                ZZ, ZZ, ZZ, ZZ, II ];
    end

    L0 = T*L0_before_T*T';

    save(fullfile(export_dir, 'L0_internal.mat'), ...
         'L0_before_T','L0','idx','-v7');

    fprintf(' Done in %.0f seconds.\n', toc);
end

