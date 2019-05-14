# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:45:29 2018\\

@author: xh
"""
#这个代码，是通过展示地图击杀和死亡最多的地方，让我们可以挑选出有哪些好玩的地方
#加载模块
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.misc.pilutil import imread


f = open(r'F:\spyder\kernel项目\pubg\PUBG_MatchData_Flattened.tsv')
df = pd.read_csv(f,sep = '\t')

#edf和mdf是两个地图，下面把两张地图分开进行处理
edf = df.loc[df['map_id'] == 'ERANGEL']
mdf = df.loc[df['map_id'] == 'MIRAMAR']

#print(edf.head())
def killer_victim_df_maker(df):
    #挑出地图中击杀和被杀玩家的坐标
    df = edf
    victim_x_df = df.filter(regex = 'victim_position_x')
    victim_y_df = df.filter(regex = 'victim_position_y')
    killer_x_df = df.filter(regex = 'killer_position_x')
    killer_y_df = df.filter(regex = 'killer_position_y')
    #ravel()将多维矩阵变成一维
    victim_x_s = pd.Series(victim_x_df.values.ravel('F'))
    victim_y_s = pd.Series(victim_y_df.values.ravel('F'))
    killer_x_s = pd.Series(killer_x_df.values.ravel('F'))
    killer_y_s = pd.Series(killer_y_df.values.ravel('F'))
    
    vdata = {'x':victim_x_s, 'y':victim_y_s}
    kdata = {'x':killer_x_s, 'y':killer_y_s}
    
    #dropna(how = 'any')删除带nan的行
    #再留下坐标等于0（在边界上的异常数据）剔除
    victim_df = pd.DataFrame(data = vdata).dropna(how = 'any')
    victim_df = victim_df[victim_df['x'] > 0]
    killer_df = pd.DataFrame(data = kdata).dropna(how = 'any')
    killer_df = killer_df[killer_df['x'] > 0]
    return killer_df, victim_df

ekdf,evdf = killer_victim_df_maker(edf)
mkdf,mvdf = killer_victim_df_maker(mdf)


# print(ekdf.head())#在森林击杀的坐标数据
# print(evdf.head())#在森林被杀的坐标数据
# print(mkdf.head())
# print(mvdf.head())
# print(len(ekdf), len(evdf), len(mkdf), len(mvdf))

#将dataframe转换成numpy array
plot_data_ev = evdf[['x','y']].values
plot_data_ek = ekdf[['x','y']].values
plot_data_mv = mvdf[['x','y']].values
plot_data_mk = mkdf[['x','y']].values

#将获得的坐标数据与地图上的坐标数据进行匹配
plot_data_ev = plot_data_ev * 4040 /800000
plot_data_ek = plot_data_ek * 4040 /800000
plot_data_mv = plot_data_mv * 976 /800000
plot_data_mk = plot_data_mk * 976 /800000

#加载模块
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize

#热力图函数
def heatmap(x, y, s, bins = 100):
#    x = plot_data_ev[:,0]
#    y = plot_data_ev[:,1]
#    s = 1.5
#    bins = 800
    #np.histogram2d()将两列数值转为矩阵
    heatmap, xedges, yedges = np.histogram2d(x, y, bins = bins)
    #高斯锐化模糊对象
    heatmap = gaussian_filter(heatmap, sigma = s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

#读取森林地图底图
#Normalize归一化
#np.clip(x,a,b)将x中小于a的值设为a，大于b的值设为b
#cm.bwr 蓝白红
bg = imread('erangel.jpg')
hmap, extent = heatmap(plot_data_ev[:,0], plot_data_ev[:,1], 1.5, bins =800)
alphas = np.clip(Normalize(0, hmap.max()/100, clip=True)(hmap)*1.5,0.0,1.)
colors = Normalize(hmap.max()/100, hmap.max()/20, clip=True)(hmap)
colors = cm.bwr(colors)
colors[..., -1] = alphas

hmap2, extent2 = heatmap(plot_data_ek[:,0],plot_data_ek[:,1],1.5, bins = 800)
alphas2 = np.clip(Normalize(0, hmap2.max()/100, clip = True)(hmap2)*1.5, 0.0, 1.)
colors2 = Normalize(hmap2.max()/100, hmap2.max()/20, clip=True)(hmap2)
colors2 = cm.RdBu(colors2)
colors2[...,-1] = alphas2

#'森林死亡率图'
fig, ax = plt.subplots(figsize = (24,24))
ax.set_xlim(0, 4096);ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent = extent, origin = 'lower', cmap = cm.bwr, alpha = 1)
#ax.imshow(colors2, extent = extent2, origin = 'lower', cmap = cm.RdBu, alpha = 0.5)
plt.gca().invert_yaxis()
plt.title('森林地图死亡率图')

#森林击杀图
fig, ax = plt.subplots(figsize = (24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors2, extent = extent2, origin = 'lower', cmap = cm.RdBu, alpha = 1)
plt.gca().invert_yaxis()
#plt.colorbar()
plt.title('森林地图击杀率图')
plt.show()

fix, ax = plt.subplots(figsize = (24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent = extent, origin = 'lower', cmap = cm.bwr, alpha = 1)
ax.imshow(colors2, extent = extent2, origin = 'lower', cmap = cm.RdBu, alpha = 1)
plt.gca().invert_yaxis()


#沙漠地图
bg = imread('miramar.jpg')
hmap, extent = heatmap(plot_data_mv[:,0], plot_data_mv[:,1], 1.5, bins = 800)
alphas = np.clip(Normalize(0, hmap.max()/200, clip=True)(hmap2)*1.5, 0.0, 1.)
colors2 = Normalize(hmap2.max()/100, hmap2.max()/20, clip=True)(hmap2)
colors2 = cm.rainbow(colors2)
colors2[..., -1] = alphas2

hmap2, extent2 = heatmap(plot_data_mv[:,0], plot_data_mv[:,1], 1.5, bins = 800)
alphas2 = np.clip(Normalize(0, hmap2.max()/200, clip=True)(hmap2)*1.5, 0.0, 1.)
colors = Normalize(hmap.max()/100, hmap.max()/20, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

a = colors2[...,-1]
colors3 = colors2
colors3[...,-1] = np.clip(abs(colors2[...,-1]-colors[...,-1]),0.0,1.)


np.mean(colors2[...,-1]-colors[...,-1])



#沙漠死亡率图
fig, ax = plt.subplots(figsize = (24,24))
ax.set_xlim(0, 1000);ax.set_ylim(0, 1000)
ax.imshow(bg)
#ax.imshow(colors, extent = extent, origin = 'lower', cmap = cm.Blues, alpha = 0.5)
ax.imshow(colors3, extent = extent2, origin = 'lower', cmap = cm.Reds, alpha = 0.5)
plt.gca().invert_yaxis()
plt.title('沙漠地图LYB图')
plt.show()
#通过对比击杀率和死亡率，寻找lyb藏身之地
#color3,击杀率大于死亡率的地方

'''
在查看上面的热图时，重要的是要记住蓝圈对整体死亡的影响。
 地图中心区域的许多紫色斑点可能是由于区域向内推动玩家！ 
 同样，通过查看时间片可以稍微减轻这种情况。 
 我将创建另一个内核来执行此操作，因为它需要更改此内核的第一个代码块。
 在此之前，让我们尝试查看每个垃圾箱中的杀死率。
 首先，让我们定义一个除法函数，这样我们就不会除以0。
'''

def divbutnotbyzero(a, b):
    c = np.zeros(a.shape)
    for i, row in enumerate(b):
        for j, el in enumerate(row):
            if el == 0:#如果击杀率等于0的话
                c[i][j] = a[i][j]#c值就等于死亡率的值
            else:
                c[i][j] = a[i][j]/el#击杀/死亡
    return c

bg = imread('erangel.jpg')
hmap, extent = heatmap(plot_data_ev[:,0], plot_data_ev[:,1], 0, bins = 800)
hmap2, extent2 = heatmap(plot_data_ek[:,0], plot_data_ek[:,1], 0, bins = 800)
hmap3 = divbutnotbyzero(hmap, hmap2)
alphas = np.clip(Normalize(0, hmap3.max()/100, clip=True)(hmap)*1.5, 0.0,1.)
colors = Normalize(hmap3.max()/100, hmap3.max()/20, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[...,-1] = alphas

fig, ax = plt.subplots(figsize = (24, 24))
ax.set_xlim(0,4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent = extent, origin = 'lower', cmap = cm.rainbow, alpha = 0.5)
plt.gca().invert_yaxis()

'''
Pretty cool! Notably, the typical "hot zones" arent
 the only places for getting a good kill/death ratio.
 Anywhere that you are seeing red is a pretty good spot to land. 
 Let's print the k/d mean:
     太酷了！ 值得注意的是，典型的“热区”并不是获得良好杀伤/死亡率的唯一场所。 
     你看到红色的任何地方都是降落的好地方。 让我们打印k / d的意思是：
'''
print(hmap3.mean())



bg = imread('miramar.jpg')
hmap, extent = heatmap(plot_data_mv[:,0], plot_data_mv[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_mk[:,0], plot_data_mk[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, hmap3.max()/100, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(hmap3.max()/100, hmap3.max()/20, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas


fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()
plt.show()













































































