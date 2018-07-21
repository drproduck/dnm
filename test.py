from Process import StickBreakingProcess
import matplotlib.pyplot as plt
stick = StickBreakingProcess(alpha=100)
sample = stick.sample(1000)
plt.hist(sample, bins=100)
plt.show()
