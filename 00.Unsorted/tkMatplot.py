# 1.step 1 of using matplotlib in tkinter
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib as mpl
mpl.use("TkAgg")


def test_plot(fig_handle):
    t = np.arange(0.0, 3, 0.01)
    s = np.sin(2 * np.pi * t)
    fig_handle.plot(t, s)


def main(wins=None):
    # ！！！用其它两个会报错
    # figUV = pl.figure(figsize=(15, 8))
    # figUV = plt.figure(figsize=(15, 8))
    # figUV = Figure(figsize=(15, 8))
    figUV = Figure(figsize=(5, 4), dpi=100)
    # 注：如果figsize 比例设置的不好，则有部分内容无法显示（ eg. 设为(15,8)，toolbar不显示）

    canvas = FigureCanvasTkAgg(figUV, master=wins)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # UV plots
    UVPlot = figUV.add_subplot(121, aspect='equal', facecolor=(0.4, 0.4, 0.4))
    UVPlot.cla()
    test_plot(UVPlot)

    # SKYUV plots
    SKYUVPlot = figUV.add_subplot(122, aspect=0.7)  # width/height
    SKYUVPlot.cla()
    test_plot(SKYUVPlot)

    # toolbox
    toolbar = NavigationToolbar2Tk(canvas, wins)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Using Matplotlib in Tkinter EXP: UV Plane")
    main(wins=root)
    tk.mainloop()
