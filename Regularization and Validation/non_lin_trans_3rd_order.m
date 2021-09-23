function [phi] = non_lin_trans_3rd_order(x)
% non_lin_trans_3rd_order: Perform a non-linear transform of the symmetry 
% and intensity features to a 3rd order polynomial set.

phi = [x(:,1), x(:,2), x(:,1).^2, x(:,1).*x(:,2), x(:,2).^2, x(:,1).^3, (x(:,1).^2).*x(:,2), x(:,1).*(x(:,2).^2), x(:,2).^3];

end

