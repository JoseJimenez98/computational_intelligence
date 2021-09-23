function [p5] = plothypothesis3(w)

syms x1 x2
eqn = w(1) + w(2)*x1 + w(3)*x2 + w(4)*x1^2 + w(5)*x1*x2 + w(6)*x2^2 + w(7)*x1^3 + w(8)*x1^2*x2 + w(9)*x1*x2^2 + w(10)*x2^3 == 0;
p5 = fimplicit(eqn,[-1 1 -1 1],'LineWidth',2,'DisplayName','Hypothesis');

end

