addpath gcs

load( 'stereo_in.mat' ); % the 'task' variable

num_pairs = size( task, 1 );

D = cell( num_pairs, 1 );

for i = 1:num_pairs
  left = task{i,1};
  right = task{i,2};
  seeds = task{i,3};

  D{i} = gcs( left, right, seeds );
  D{i} = - D{i}; % the gcs returns disparities with opposite sign

  figure
  imagesc( D{i} )
%   drawnow
end


save( 'stereo_out.mat', 'D' )