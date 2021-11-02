% DEMO_P5  Verification of the p5 algorithm
% (it works for every non-degenerated 5-tuple of correspondences)

% (c) 2010-10-19, Martin Matousek
% Last change: $Date::                            $
%              $Revision$


if( ~exist( 'p5gb', 'file' ) )
  error( 'Cannot find five-point estimator. Probably PATH is not set.' );
end


u1 = randn( 2, 5 )*10;
u2 = randn( 2, 5 );

u1p = [ u1; ones( 1, 5 ) ];
u2p = [ u2; ones( 1, 5 ) ];

Es = p5gb( u1p, u2p );

fprintf( '%15s %15s %15s\n', 'det(E)', 'max alg err','max geom err' );

R1 = {};
R2 = {};
t = {};
for i = 1:size(Es,2)
  E = reshape( Es(:,i), 3, 3 )'; % row-ordered !
  E = E / norm( E, 'fro' );  % just to work with reasonable numbers
  
  % algebraic err. - err. of the equation   u2p' * E * u1p = 0
  alg_err = diag( u2p' * E * u1p ); 

  % geometric err. - distance of u2 to epipolar line corresponding to u1.
  
  l2 = E * u1p;  % epipolar lines 
  
  % distances, normalization by line's normal vectors (u2p allready normalized)
  geom_err = sum( u2p .* l2 ) ./  sqrt( l2(1,:).^2 + l2(2,:).^2 );
  
  
  fprintf( '%15g %15g %15g\n', ...
           det(E), max( abs( alg_err ) ), max( abs( geom_err ) ) );
end
