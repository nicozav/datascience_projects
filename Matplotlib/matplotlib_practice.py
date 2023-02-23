#Run Cell
#%%

from cProfile import label
from symbol import dotted_as_name
from turtle import color
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np

x = np.linspace(0,5,11)
y = x**2
# %%

#Functional Method Practice

plt.plot(x,y)
# %%
plt.xlabel('Test Label X')
plt.ylabel('Test Label Y')
plt.title('Matplotlib Practice Graph')
# %%
plt.plot(x,y)
plt.xlabel('Test Label X')
plt.ylabel('Test Label Y')
plt.title('Matplotlib Practice Graph')
# %%
plt.subplot(1,2,1)
plt.plot(x,y,'r')
plt.title('Red Graph')

plt.subplot(1,2,2)
plt.plot(y,x,'b')
plt.title('Blue Graph')
# %%
# Object Oriented Graphs

figure_1 = plt.figure()

graph_1 = figure_1.add_axes([0.1,0.1,0.8,0.8])

graph_1.plot(x,y,'p')
graph_1.set_title('Object Oriented Graph Testing')
graph_1.set_xlabel('Test Label X')
graph_1.set_ylabel('Test Label Y')

# %%
fig,subplots_1 = plt.subplots(1,2)

subplots_1[0].plot(x,y,'r')
subplots_1[1].plot(y,x,'b')


plt.tight_layout()
# %%
figure_2,graph_2 = plt.subplots(nrows=2,ncols=1,figsize=(8,4))

graph_2[0].plot(x,y,'r')
graph_2[0].set_title('Red Graph')

graph_2[1].plot(y,x,'b')
graph_2[1].set_title('Blue Graph')


plt.tight_layout()
# %%
figure_2.savefig('/Users/nicholaszavala/Documents/Test Picture',dpi=300)
# %%
figure_3 = plt.figure()

graph_3 = figure_3.add_axes([0,0,1,1])

graph_3.set_title('Testing Title for 3 lines')
graph_3.set_xlabel('Test Label X')
graph_3.set_ylabel('Test Label Y')

graph_3.plot(x,y,label='X Normal',color='black', linewidth= 2, linestyle = ':')
graph_3.plot(x,y*2,label='X times 2',color='pink', linewidth = 2, linestyle = '-.')
graph_3.plot(x,y*3,label='X times 3',color='purple', linewidth = 2, marker = 'x')

graph_3.set_xlim(0,5)
graph_3.set_ylim(0,5)

graph_3.legend()

plt.tight_layout()
# %%
