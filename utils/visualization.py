# utils/visualization.py
import matplotlib.pyplot as plt

def plot_training_curve(x, ys, labels, xlabel, ylabel, title=None):
    plt.figure(figsize=(8,4))
    for y,lab in zip(ys,labels): plt.plot(x,y,label=lab)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if title: plt.title(title)
    plt.legend(); plt.grid(True)
    plt.show()

