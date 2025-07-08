import sympy as sp

x, y = sp.symbols('x y', real=True)
df = sp.Symbol('d_f', positive=True, real=True)
g = sp.Symbol('g', positive=True)
R = sp.Symbol('R', real=True)

phi = sp.Function('phi')(x)
phi0 = sp.Symbol('phi0', positive=True, real=True)
m, lam, zeta, kappa, gamma = sp.symbols('m lambda zeta kappa gamma', real=True, positive=True)

L_local = (1/2)*sp.Derivative(phi, x)**2 + (m**2)/2 * phi**2 + lam/24 * phi**4
L_curv = zeta * (df - 1) * R * phi**2
L_log = gamma * phi**2 * sp.log(phi**2 / phi0**2)
L_total = sp.sqrt(g) * (L_local + L_curv + L_log)

print("\nLagrangiana efectiva (local):")
sp.pprint(sp.simplify(L_total))

print("\nTérmino no local (formato formal):")
print("S_nonlocal = kappa ∬ [phi²(x) * phi²(y)] / |x - y|^{d_f + 2} dx dy")
