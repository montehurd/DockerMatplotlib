#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

import base64
import io
from matplotlib import animation
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.tri as tri
import matplotlib.animation as animation
from time import sleep
import pandas as pd

def imageTagForData(data, extension):
  return '<img src="data:image/{0};base64, {1}">'.format(extension, base64.b64encode(data).decode())


def base64PNGImageTagForPlot(plot):
  buffer = io.BytesIO()
  plot.savefig(buffer, format='png')
  buffer.seek(0)
  pngData = buffer.read()
  buffer.close()
  return imageTagForData(pngData, 'png')


def base64GIFImageTagForAnimation(animation):
  with NamedTemporaryFile(suffix='.gif') as file:
    animation.save(file.name, writer='pillow', fps=15)
    gifData = open(file.name, "rb").read()
  return imageTagForData(gifData, 'gif')


print('''
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <style>
      img, div {
        margin: 0 auto;
        display: block;
      }
      div {
        width: 820px;
      }
      body {
        padding: 20px;
      }
    </style>
  </head>
  <body>
    <div>
''')


plt.clf()
print('Example 1')
objects = ('Ant', 'Fish', 'Bat', 'Rat', 'Cat', 'Goat')
y_pos = np.arange(len(objects))
performance = [1,2,4,6,8,10]
plt.margins(.03)
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Numbers')
plt.title('Animals')
plt.ylim(top=11)
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 2')
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 3')
x = np.linspace(0, 2, 100)
fig, ax = plt.subplots()
ax.plot(x, x, label='linear')
ax.plot(x, x**2, label='quadratic')
ax.plot(x, x**3, label='cubic')
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_title("Simple Plot")
ax.legend()
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 4')
# Fixing random state for reproducibility
np.random.seed(19680801)
dt = 0.01
t = np.arange(0, 30, dt)
nse1 = np.random.randn(len(t))                 # white noise 1
nse2 = np.random.randn(len(t))                 # white noise 2
# Two signals with a coherent part at 10Hz and a random part
s1 = np.sin(2 * np.pi * 10 * t) + nse1
s2 = np.sin(2 * np.pi * 10 * t) + nse2
fig, axs = plt.subplots(2, 1)
axs[0].plot(t, s1, t, s2)
axs[0].set_xlim(0, 2)
axs[0].set_xlabel('time')
axs[0].set_ylabel('s1 and s2')
axs[0].grid(True)
cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
axs[1].set_ylabel('coherence')
fig.set_tight_layout(True)
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 5')
x = np.linspace(-np.pi/2, np.pi/2, 31)
y = np.cos(x)**3
# 1) remove points where y > 0.7
x2 = x[y <= 0.7]
y2 = y[y <= 0.7]
# 2) mask points where y > 0.7
y3 = np.ma.masked_where(y > 0.7, y)
# 3) set to NaN where y > 0.7
y4 = y.copy()
y4[y3 > 0.7] = np.nan
plt.plot(x*0.1, y, 'o-', color='lightgrey', label='No mask')
plt.plot(x2*0.4, y2, 'o-', label='Points removed')
plt.plot(x*0.7, y3, 'o-', label='Masked values')
plt.plot(x*1.0, y4, 'o-', label='NaN values')
plt.legend()
plt.title('Masked and NaN data')
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 6')
points = np.ones(5)  # Draw 5 points for each line
marker_style = dict(color='blue', linestyle=':', marker='o', markersize=15, markerfacecoloralt='red')
fig, ax = plt.subplots()
fig.set_tight_layout(False)
# Plot all fill styles.
for y, fill_style in enumerate(Line2D.fillStyles):
  ax.text(-0.5, y, repr(fill_style), horizontalalignment='center', verticalalignment='center')
  ax.plot(y * points, fillstyle=fill_style, **marker_style)
ax.set_axis_off()
ax.set_title('fill style')
ax.margins(.03)
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 7')
arr = np.arange(100).reshape((10, 10))
plt.close('all')
fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)
im = ax.imshow(arr, interpolation="none")
fig.set_tight_layout(True)
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 8')
np.random.seed(19680801)
fig, ax = plt.subplots()
for color in ['blue', 'orange', 'green']:
  n = 750
  x, y = np.random.rand(2, n)
  scale = 200.0 * np.random.rand(n)
  ax.scatter(x, y, c=color, s=scale, label=color, alpha=0.3, edgecolors='none')
ax.legend()
ax.grid(True)
plt.margins(0.05)
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 9')
plt.margins(0.05)
x = np.linspace(0.1, 2 * np.pi, 41)
y = np.exp(np.sin(x))
plt.stem(x, y, use_line_collection=True)
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 10')
# Fixing random state for reproducibility
np.random.seed(19680801)
# the bar
x = np.random.rand(500) > 0.7
barprops = dict(aspect='auto', cmap='binary', interpolation='nearest')
fig = plt.figure()
# a vertical barcode
ax1 = fig.add_axes([0.1, 0.1, 0.1, 0.8])
ax1.set_axis_off()
ax1.imshow(x.reshape((-1, 1)), **barprops)
# a horizontal barcode
ax2 = fig.add_axes([0.3, 0.4, 0.6, 0.2])
ax2.set_axis_off()
ax2.imshow(x.reshape((1, -1)), **barprops)
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 11')
N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, menMeans, width, yerr=menStd)
p2 = plt.bar(ind, womenMeans, width, bottom=menMeans, yerr=womenStd, color='orange')
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 12')
delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Simplest default with labels')
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 13')
X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
U, V = np.meshgrid(X, Y)
fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
ax.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 14')
Z = np.random.rand(6, 10)
fig, (ax0, ax1) = plt.subplots(2, 1)
c = ax0.pcolor(Z)
ax0.set_title('default: no edges')
c = ax1.pcolor(Z, edgecolors='k', linewidths=4)
ax1.set_title('thick edges')
fig.set_tight_layout(True)
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 15')
plt.figure()
plt.subplot(111, projection="aitoff")
plt.title("Aitoff")
plt.grid(True)
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 16')
# Fixing random state for reproducibility
np.random.seed(19680801)
n = 100000
x = np.random.standard_normal(n)
y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
ax = axs[0]
hb = ax.hexbin(x, y, gridsize=50, cmap='terrain')
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax.set_title("Hexagon binning")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('counts')
ax = axs[1]
hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap='terrain')
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax.set_title("With a log color scale")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')
print(base64PNGImageTagForPlot(plt))


plt.clf()
print('Example 17')
#-----------------------------------------------------------------------------
# Analytical test function
#-----------------------------------------------------------------------------
def function_z(x, y):
    r1 = np.sqrt((0.5 - x)**2 + (0.5 - y)**2)
    theta1 = np.arctan2(0.5 - x, 0.5 - y)
    r2 = np.sqrt((-x - 0.2)**2 + (-y - 0.2)**2)
    theta2 = np.arctan2(-x - 0.2, -y - 0.2)
    z = -(2 * (np.exp((r1 / 10)**2) - 1) * 30. * np.cos(7. * theta1) +
          (np.exp((r2 / 10)**2) - 1) * 30. * np.cos(11. * theta2) +
          0.7 * (x**2 + y**2))
    return (np.max(z) - z) / (np.max(z) - np.min(z))
#-----------------------------------------------------------------------------
# Creating a Triangulation
#-----------------------------------------------------------------------------
# First create the x and y coordinates of the points.
n_angles = 20
n_radii = 10
min_radius = 0.15
radii = np.linspace(min_radius, 0.95, n_radii)
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += np.pi / n_angles
x = (radii * np.cos(angles)).flatten()
y = (radii * np.sin(angles)).flatten()
z = function_z(x, y)
# Now create the Triangulation.
# (Creating a Triangulation without specifying the triangles results in the
# Delaunay triangulation of the points.)
triang = tri.Triangulation(x, y)
# Mask off unwanted triangles.
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                         y[triang.triangles].mean(axis=1))
                < min_radius)
#-----------------------------------------------------------------------------
# Refine data
#-----------------------------------------------------------------------------
refiner = tri.UniformTriRefiner(triang)
tri_refi, z_test_refi = refiner.refine_field(z, subdiv=3)
#-----------------------------------------------------------------------------
# Plot the triangulation and the high-res iso-contours
#-----------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.triplot(triang, lw=0.5, color='white')
levels = np.arange(0., 1., 0.025)
cmap = cm.get_cmap(name='terrain', lut=None)
ax.tricontourf(tri_refi, z_test_refi, levels=levels, cmap=cmap)
ax.tricontour(tri_refi, z_test_refi, levels=levels,
               colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
               linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])
ax.set_title("High-resolution tricontouring")
print(base64PNGImageTagForPlot(plt))


print('<br><br><hr>The three animation examples below take a while to load - have not optimized them at all :) The third one may take a full minute to appear - it crunches a huge data set.<hr><br><br>')


plt.clf()
print('Example 18')
TWOPI = 2*np.pi
fig, ax = plt.subplots()
t = np.arange(0.0, TWOPI, 0.001)
s = np.sin(t)
l = plt.plot(t, s)
ax = plt.axis([0,TWOPI,-1,1])
redDot, = plt.plot([0], [np.sin(0)], 'ro')
def animate(i):
  redDot.set_data(i, np.sin(i))
  return redDot,
anim = animation.FuncAnimation(fig, animate, frames=np.arange(0.0, TWOPI, 0.1), interval=10, repeat=False)
print(base64GIFImageTagForAnimation(anim))


plt.clf()
print('Example 19')
fig = plt.figure()
def f(x, y):
  return np.sin(x) + np.cos(y)
x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
  x += np.pi / 15.
  y += np.pi / 20.
  im = plt.imshow(f(x, y), animated=True)
  ims.append([im])
anim = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
print(base64GIFImageTagForAnimation(anim))


plt.clf()
print('Example 20')
# This example based on: https://opensource.com/article/20/4/python-data-covid-19
#### ---- Step 1:- Download data
URL_DATASET = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
df = pd.read_csv(URL_DATASET, usecols = ['Date', 'Country', 'Confirmed'])

#### ---- Step 2:- Create list of all dates
list_dates = df['Date'].unique()
# print(list_dates) # Uncomment to see the dates

#### --- Step 3:- Pick 5 countries. Also create ax object
fig, ax = plt.subplots(figsize=(8, 4))
# We will animate for these 5 countries only
list_countries = ['India', 'China', 'US', 'Italy', 'Spain']
# colors for the 5 horizontal bars
list_colors = ['black', 'red', 'green', 'blue', 'yellow']

### --- Step 4:- Write the call back function
# plot_bar() is the call back function used in FuncAnimation class object
def plot_bar(some_date):
    df2 = df[df['Date'].eq(some_date)]
    ax.clear()
    # Only take Confirmed column in descending order
    df3 = df2.sort_values(by = 'Confirmed', ascending = False)
    # Select the top 5 Confirmed countries
    df4 = df3[df3['Country'].isin(list_countries)]
    # print(df4)  # Uncomment to see that dat is only for 5 countries
    sleep(0.2)  # To slow down the animation
    # ax.barh() makes a horizontal bar plot.
    return ax.barh(df4['Country'], df4['Confirmed'], color= list_colors)

###----Step 5:- Create FuncAnimation object---------
my_anim = animation.FuncAnimation(fig = fig, func = plot_bar, frames= list_dates, blit=True, interval=20)
print(base64GIFImageTagForAnimation(my_anim))


print('''
    </div>
  </body>
</html>
''')
