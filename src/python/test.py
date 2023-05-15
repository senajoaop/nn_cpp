import sys
import sympy
import numpy as np 
from sympy import Pow, Rational, Mul, Add, Integer
import matplotlib.pyplot as plt

plt.style.use('seaborn')





val = []
with open("/../data/data_vec.txt", 'r') as file:
    for line in file:
        data = line.split(',')
        # data = list(filter(None, data))
        # data = list(filter(lambda x: '\n', data))
        data = [item for item in data if item!='']
        data = [item for item in data if item!='\n']
        val.append(np.array([float(item) for item in data]))


print(val[0].shape, val[0])

for i in range(1, 40):
    plt.plot(val[0], val[i], label=f"R={i}")

# plt.plot(val[0], val[1])

plt.legend()
plt.show()










sys.exit()

a = 3
b = 7

X = np.linspace(-100, 100, 1000)
X2 = X - a
Y = (X2)**2+a

a = np.linspace(-4, 4, 10)
for item in a:
    # X2 = X - item
    # Y = (X2)**2+item
    X2 = X-item
    Y = (X2-b)**2 + (X-b)*item +b
    plt.plot(X, Y)

plt.xlim(-100, 100)
plt.ylim(-100, 100)

# plt.show()


# sys.exit()


X = np.linspace(-10, 10, 1000)
Y = np.linspace(-10, 10, 1000)
Y2 = (Y**2)
X2 = X - Y

X2, Y2 = np.meshgrid(X2, Y2)

Z = (X2)**2+(Y2**2)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()



# plt.plot(X, Y)
#
#
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# # plt.grid()
#
#
# plt.show()





# sys.exit()

x, c1, c2, c3, c4, c5, c6 = sympy.symbols("x c1 c2 c3 c4 c5 c6")

eq = (c1 - c2*(c3/(1/x + c4 + c5)))*c6

deq = sympy.diff(eq, x)

sympy.pprint(eq)
sympy.pprint(deq)


sys.exit()

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

