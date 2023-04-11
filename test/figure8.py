straightness = [0.09, 0.1412, 0.1541, 0.1676, 0.1798, 0.1812, 0.2015, 0.2151, 0.2315, 0.2291, 0.2281, 0.2295]
error = [2.6, 2.23, 1.61, 1.41, 1.13, 0.97, 0.96, 0.89, 0.91, 0.91, 0.92, 0.90, 0.91]
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)

twin1 = ax.twinx()
p1, = ax.plot(straightness, "b-", label="straightness")
p2, = twin1.plot(error, "r-", label="percentage error(%)")
ax.set_xlabel("number of initializing objects")
ax.set_ylabel("straightness")
twin1.set_ylabel("percentage error(%)")

ax.legend(handles=[p1, p2], loc='center right')

plt.show()