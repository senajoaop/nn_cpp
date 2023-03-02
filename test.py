import sympy
import numpy as np 
from sympy import Pow, Rational, Mul, Add, Integer




m, L, hr, x = sympy.symbols("m L hr x")


sig = (sympy.sin(hr) - sympy.sinh(hr)) / (sympy.cos(hr) + sympy.cosh(hr))

sig = Mul(Add(sympy.sin(hr), Mul(sympy.sinh(hr), Integer(-1))), Pow(Add(sympy.cos(hr), sympy.cosh(hr)), Integer(-1)))

sig = sympy.symbols("sigma_r")

# print(Rational(1, m))

term1 = Pow(Pow(Mul(m, L), Integer(-1)), Rational(1, 2))
term2 = sympy.cos(Mul(hr, x, Pow(L, Integer(-1))))
term3 = sympy.cosh(Mul(hr, x, Pow(L, Integer(-1))))
term4 = sympy.sin(Mul(hr, x, Pow(L, Integer(-1))))
term5 = sympy.sinh(Mul(hr, x, Pow(L, Integer(-1))))

# print(term)

phi = Mul(term1, Add(Add(term2, Mul(term3, Integer(-1))), Mul(sig, Add(term4, Mul(term5, Integer(-1))))))


dphidx = sympy.diff(phi, x)

print(dphidx)
print(sympy.ccode(dphidx))

