import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(0, 4.0, 10000)
x = 1e-5 * np.ones(10000)


plt.figure(figsize=(12, 6))
for i in range(1000):
    x = r * x * (1 - x)
    if i > 950:
        plt.plot(r, x, ',k', alpha=0.25)

plt.title("Bifurcation Diagram (Feigenbaum Constant Visualized)")
plt.xlabel("r")
plt.ylabel("x")
plt.grid(True)
plt.show()