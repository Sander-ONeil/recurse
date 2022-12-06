import numpy as np
from numba import jit, guvectorize, vectorize, float64
import pygame
import time
clock = pygame.time.Clock()
width=1400
height=1400
print('Resolution : '+str(width)+ " " + str(height))
sca=1
screen = pygame.display.set_mode((width*sca,height*sca))
clock = pygame.time.Clock()
update,done=1,False

from quats import *

def cell():
    return vec([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[1,0,0],[0,0,1],[-1,0,0],[0,0,-1],[0,1,0],[0,0,1],[0,-1,0],[0,0,-1],[0,-1,0]])

def show(vp,co = (255,0,0)):
    
    Xs = vp[0]*1+vp[1]*0+vp[2]*0 + width/2
    Ys = -(vp[0]*0+vp[1]*1+vp[2]*0) + height/2

    
    C = np.stack((Xs,Ys),axis = -1)
    layersns = [fractal[f][0]for f in range(fractal_layers)]
    
    tsts = 1
    for s in layersns:
        tsts*=s
    #print(tsts)
    #record(4)
    record(4)
    i=0
    drqw = np.zeros((tsts*C.shape[0],2))
    for index in np.ndindex(tuple(layersns)):
        
        #print(index)
        realindex = tuple([slice(C.shape[0])] + list(index) + [slice(2)])
        #print(realindex)
        
        
        ps = C[realindex]
        record(5)
        
        #print(ps)
        #pygame.draw.lines(screen,((i/200)%155+100,(i*20)%155+100,(i)%155+100),False,ps,2)
        pygame.draw.polygon(screen,((i*3)%155+100,(i*20)%155+100,(i*200)%155+100),ps)
        
        #print(ps.shape)
        #print(i)
        drqw[i:i+ps.shape[0],:] = ps
        
        i+=ps.shape[0]
    
    #pygame.draw.lines(screen,(255,255,255),False,drqw,2)
    record(6)
    #print(i)
    
    

v = (vec([[1,0,0],[0,0,0],[.5,np.sin(np.pi/3),0],[1,0,0]])-vec3(.5,np.sin(np.pi/3)*1/3,0))
#v = cell()*20

#v = np.zeros((15,3))
#v = np.array([[0,1,0],[0,0,0]],dtype = np.float64)

# for x in range(15):
#     v[x] = [np.cos(x*np.pi/15*2),np.sin(x*np.pi/15*2),0]

v = np.transpose(v)
up = vec([0,1,0])
le = vec([1,0,0])
fo = vec([0,0,1])

v = v.reshape((3,v.shape[1]))
    

t = 0
frames= 0
mousedown=False


fractal_layers=4
n = 6
#fractal   iters rotation displacement scale
fractal = [[n*1,defaultq(),vec3(0,0,0),1,0]for f in range(fractal_layers)]
#fractal[0][0] = 10

global controlled_layer,controlled_func
controlled_layer = 0

controlled_func = 1
funcs=3


times = ['beg1','ds','scs','qt','draw1','draw2','draw3','end']

global tl
tl = time.time()
TS = [0,0,0,0,0,0,0,0]
def record(n):

    global tl
    TS[n]+= time.time()-tl
    tl = time.time()


def changefunc():
    global controlled_func
    controlled_func+=1
    controlled_func%=funcs
def changefuncd():
    global controlled_func
    controlled_func=1
def changefuncr():
    global controlled_func
    controlled_func=0
def changefuncs():
    global controlled_func
    controlled_func=2
def changelayer():
    global controlled_layer
    controlled_layer+=1
    controlled_layer%=fractal_layers
def resvar():
    clean = [n*1,defaultq(g)+0,vec3(0,0,0),1]
    fractal[controlled_layer][controlled_func+1]=clean[controlled_func+1]

import button

button.buts = button.buttons([
   

    button.button( 10,440,'Next func',changefunc),
    button.button( 10,540,'displace',changefuncd),
    button.button( 10,640,'rotate',changefuncr),
    button.button( 10,740,'scale',changefuncs),
    button.button( 10,840,'Next layer',changelayer),
    button.button( 10,940,'resest current variable',resvar),

    ])



while not done:
    record(0)
    dt = clock.tick(500)/1000
    t +=dt
    frames +=1
    m = (vec(pygame.mouse.get_pos())-vec([width,height])/2)*vec([1,-1])
    mousepos = vec3(m[0],m[1],0)
    #z = np.sqrt(500-np.linalg.norm(mousepos))
    M = na(mousepos+vec3(0,0,1)+0)
    
    g = lookats(M,face=vec3(1,0,0))
    g2 = lookats(M,face=vec3(0,1,0))

    M = 10
    
    layersns = [fractal[f][0]for f in range(fractal_layers)]
    #print(layersns)
    ds = np.zeros(([3]+layersns))
    qt = np.zeros(([4]+layersns))
    qt[0] = 1
    scs = np.ones(layersns)
    
    

    s=1
    for f in fractal:
        q = f[1]
        qr = q[0]
        qi = q[1]
        qj = q[2]
        qk = q[3]
        
        f[4] = vec([
        [ 1 - 2*(qj*qj+qk*qk), 2*(qi*qj-qk*qr), 2*(qi*qk+qj*qr)],
        
        [ 2*(qi*qj+qk*qr), 1 - 2*(qi*qi+qk*qk), 2*(qj*qk-qi*qr)],
        
        [ 2*(qi*qk - qj*qr), 2*(qj*qk+qi*qr), 1 - 2*(qi*qi+qj*qj)]
        ])
    
    
    for l in range(fractal_layers):
        sc = fractal[l][3]
        d = fractal[l][2]
        q = fractal[l][1]
        R = fractal[l][4]
        
        index1 = tuple([slice(3)]+[0]*(fractal_layers-l-1)+[0])
        index2 = tuple([0]*(fractal_layers-l-1)+[0])
        index3 = tuple([slice(4)]+[0]*(fractal_layers-l-1)+[0])
        indexd = tuple([slice(3)]+[None]*(l))
        
        d= d[indexd]
        
        ld = ds[index1]
        
        ls = scs[index2]
        
        lq = qt[index3]
        
        for z in range(1,fractal[l][0]):
            
            index1 = tuple([slice(3)]+[0]*(fractal_layers-l-1)+[z])
            
            index2 = tuple([0]*(fractal_layers-l-1)+[z])
            
            index3 = tuple([slice(4)]+[0]*(fractal_layers-l-1)+[z])
            
            record(0)
            #print(indexd)
            
            ds[index1] = ld = sc*(dotvec(ld,R)+d)
            record(1)
            
            scs[index2] = ls = sc*ls
            record(2)
            
            qt[index3] = lq = quatmult(lq,q)
            record(3)

    v_enlarged = v.reshape(list(v.shape)+[1]*fractal_layers)
    v_enlarged = np.tile(v_enlarged,[1,1]+layersns)
    
    #print(qt)
    vp = 100*(scs*rot(v_enlarged,qt)+ds[:,np.newaxis])
    
    
    show(vp)
    
    button.buts.update(screen)
    
    pygame.display.flip()
    screen.fill((0,0,0))

    
    for ev in pygame.event.get():
        if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    done=True
                if ev.key == pygame.K_e:
                    controlled_layer+=1
                    controlled_layer%=fractal_layers
                if ev.key == pygame.K_c:
                    controlled_func+=1
                    controlled_func%=funcs
        if ev.type == pygame.MOUSEBUTTONDOWN:
            if not button.buts.check_pressed():
                mousedown=True
        if ev.type == pygame.MOUSEBUTTONUP:
            mousedown=False
    
    if mousedown:
        if controlled_func == 1:
                fractal[controlled_layer][2] = mousepos/100
        elif controlled_func == 0:
            fractal[controlled_layer][1] = g+0
        elif controlled_func == 2:
            fractal[controlled_layer][3] = m[0]/width*2+1
    record(7)
    

    
print(frames/t)
FT=0
for x in range(len(TS)):
    print(times[x],frames/TS[x],end='  ')
    print('fraction: ',TS[x]/t)
    print('')
    FT+=TS[x]/t
print(FT)
