
function BF = readbaseflow(mesh,BF)
    
    fprintf( '--- Reading base flow ...  '); tic();
    load BaseFlows/base_flow; %reads 'data' from mat file
    
    % Sets up base flow fields
    X = mesh.X;
    Y = mesh.Y;
    
    BF.RHO      = rho.';    % Baseflow Density 
    BF.U        = u.';      % Baseflow U velocity
    BF.V        = v.';      % Baseflow V velocity
    BF.W        = w.';      % Baseflow W velocity
    BF.T        = T.';      % Baseflow Temperature
    BF.MU       = X*0+1;    % Baseflow viscosity 
    BF.cv       = 1/(BF.kappa*(BF.kappa-1)*BF.Ma^2); % ideal gas ???
    BF.c1       = (BF.kappa-1)*BF.Re*BF.Pr*BF.Ma^2;  % ideal gas ???
    BF.c2       = BF.kappa*BF.Ma^2;
    
    % derivatives of the viscosity can be provided here, or overwriten
    % latter using, e.g., the sutherland_air function.
    BF.dMUdT    = X*0;      % Baseflow viscosity variation with temperature 
    BF.d2MUdT2  = X*0;      % Baseflow viscosity 2nd derivative wrt temperature
    BF.kappa    = 1.4;      % ideal gas adiabatic coefficient

    % Interpolate W and V velocities from database.
    %BF.U(:) = griddata(x_BF(:),y_BF(:),U_BF(:),X(:),Y(:),'cubic'); 
    %BF.V(:) = griddata(x_BF(:),y_BF(:),V_BF(:),X(:),Y(:),'cubic'); 

    fprintf( ' Done in %.0f seconds.\n',toc); 

  
    