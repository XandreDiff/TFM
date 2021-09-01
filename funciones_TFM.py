# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:42:21 2021

@author: Xandre
"""

"""

    Paquetes utilizados para cargado de datos y definición de funciones

"""

import numpy as np # numpy
import matplotlib.pyplot as plt # matplotlib pyplot
import os # para ver y cambiar directorio
import glob # abrir archivos en una carpeta
import time # da el tiempo de inicio (ordenador)

"""

    FUNCIONES PARA OBTENER LOS PUNTOS DE RETORNO A PARTIR DE n0 Y n (Metricon)
    
"""

def whiteTE(n0, n, nc = 1 ): # nc del aire por defecto
    
    """ 
        Reconstruye los puntos de retorno z suponiendo que el perfil n(x)
        se construye linealmente a partir de los indices efectivos n[i]
        y teniendo en cuenta luz TE
    """
    
    nef = (np.zeros(len(n)+1)).tolist()
    nef[0] = np.float64(n0)
    nef[1:len(nef)] = n[:]
    
    
    i, j = 0, 0
    z = (np.zeros(len(nef))).tolist() # vector de zeros
    z[0] = np.float64(0) # punto de retorno superficial
    z[1] = ((3.0/2.0)*((nef[0]+3.0*nef[1])/2.0)**(-0.5)*(nef[0]-nef[1])**(-0.5)*(0.125 
            + (1/(2.0*np.pi))*np.arctan((((nef[1])**2-nc**2)/((nef[0])**2-(nef[1])**2))**(0.5))))
    
    # punto de retorno 1 TE
    for i in range(2,len(nef)): # desde m = 2
    
        s = 0
        sumas = [] # para poder hacer el sumatorio hasta m-1, vector de sumandos
        
        for j in range(1,i):  
            
            a = (((nef[j-1]+nef[j])/2.0 + nef[i])**(0.5)*((z[j]-z[j-1])/(nef[j-1]-nef[j]))
                 *((nef[j-1]-nef[i])**(1.5)-(nef[j]-nef[i])**(1.5)))
            sumas.append(a)
            
        s = sum(sumas)  # suma de los sumandos para cada j
        
        z[i] = (z[i-1] + ((3.0/2.0)*((nef[i-1]+3.0*nef[i])/2.0)**(-0.5)*(nef[i-1]-nef[i])**(-0.5))
                *((4.0*i -3)/8 + (1/(2.0*np.pi))
                  *np.arctan((((nef[i])**2-nc**2)/((nef[0])**2-(nef[i])**2))**(0.5))- (2.0/3.0)*s))
   
        # puntos de retorno TE hasta M-2 
    lmda = 0.6328 # unidades um
    z = [i * lmda for i in z] # devuelve el valor reescalado por lambda
    return z,nef 

def whiteTM(n0, n, nc = 1): # nc del aire por defecto
    
    """ 
        Reconstruye los puntos de retorno z suponiendo que el perfil n(x)
        se construye linealmente a partir de los indices efectivos n[i]
        para luz TM
    """
    
    nef = (np.zeros(len(n)+1)).tolist()
    nef[0] = np.float64(n0)
    nef[1:len(nef)] = n[:]
    a_m = (nef[0]/nc)**4  # factor de asimetría TM
    
    
    i, j = 0,0
    z = (np.zeros(len(nef))).tolist() # vector de zeros
    z[0] = np.float64(0)
    z[1] = ((3.0/2.0)*((nef[0]+3.0*nef[1])/2.0)**(-0.5)*(nef[0]-nef[1])**(-0.5)*(0.125 
           + (1/(2.0*np.pi))*np.arctan(a_m*(((nef[1])**2-nc**2)/((nef[0])**2-(nef[1])**2))**(0.5))))
    
    for i in range(2,len(nef)): # desde m = 2
    
        s = 0
        sumas = [] # para poder hacer el sumatorio hasta m-1, vector de sumandos
        
        for j in range(1,i):  
            
            a = (((nef[j-1]+nef[j])/2.0 + nef[i])**(0.5)*((z[j]-z[j-1])/(nef[j-1]-nef[j]))
                 *((nef[j-1]-nef[i])**(1.5)-(nef[j]-nef[i])**(1.5)))
            sumas.append(a)
            
        s = sum(sumas)  # suma de los sumandos para cada j  
        
        z[i] = (z[i-1] + ((3.0/2.0)*((nef[i-1]+3.0*nef[i])/2.0)**(-0.5)*(nef[i-1]-nef[i])**(-0.5))
                *((4.0*i -3)/8 + (1/(2.0*np.pi))
                  *np.arctan(a_m*(((nef[i])**2-nc**2)/((nef[0])**2-(nef[i])**2))**(0.5))
                  -(2.0/3.0)*s))
        # puntos de retorno TM hasta M-2 
    lmda = 0.6328 # unidades um 
    z = [i * lmda for i in z] # devuelve el valor reescalado por lambda
    return z,nef

'''
    FUNCIÓN PARA MINIMIZAR n0:  AREAS DE TRIÁNGULOS CON DERIVADAS
'''

def minimozip(l1,l2):
    
    """ 
        Para cada par de puntos (funcion,n0) selecciona la pareja que haga
        funcion minimal
    """
    
    zippedlist = list(zip(l1,l2))
    return sorted(zippedlist,reverse = False, key = lambda x:x[1])[0]

def derh(z, deltanef): # tambien puede ser neff
    
    """
        Hace la derivada con espaciado constante dz/d(deltan) (se reaprovechará despues en B-M)
    """
    
    dz = (np.zeros(len(z))).tolist()
    
    dz[0] = 0.5*(-3.0*z[0]+4.0*z[1]-z[2]) # esquema a dos vecinos para delante en el extremo
    dz[len(z)-1] = 0.5*(z[len(z)-3]-4.0*z[len(z)-2]+3.0*z[len(z)-1]) # idem a vecinos para atrás
    
    for i in range(1,len(z)-1):
        dz[i] = 0.5*(z[i+1]-z[i-1]) # esquema de diferencias centrales
        
    # lo mismo para dn
    
    dn = (np.zeros(len(deltanef))).tolist()
    
    dn[0] = 0.5*(-3.0*deltanef[0]+4.0*deltanef[1]-deltanef[2])
    dn[len(deltanef)-1] = (0.5*(deltanef[len(deltanef)-3]
                                -4.0*deltanef[len(deltanef)-2]+3.0*deltanef[len(deltanef)-1]))
    
    for i in range(1,len(deltanef)-1):
        dn[i] = 0.5*(deltanef[i+1]-deltanef[i-1])
    
    dzdn = [dz[i]/dn[i] for i in range(len(z))]
    return dzdn

def areaTEder(n0,n):
    
    """
        Calculo de la suma de las areas de los tríangulos cuyos vértices vienen
        dados por las coordenadas (z[i],dn[i]/dz[i]) para TE  
    """
    
    z, nef = whiteTE(n0, n)
    dndz = derh(nef,z)
    index=len(nef)-2
    ar = (np.zeros(index)).tolist()

    for i in range(index):
        ar[i] = ((z[i]*(dndz[i+1]-dndz[i+2])+z[i+1]*(dndz[i+2]-dndz[i])
                  +z[i+2]*(dndz[i]-dndz[i+1])))/2
    return sum(ar)

def areaTMder(n0,n):
    
    """
        Calculo de la suma de las areas de los tríangulos cuyos vértices vienen
        dados por las coordenadas (z[i],dn[i]/dz[i]) para TM 
    """
    
    z, nef = whiteTM(n0, n)
    dndz = derh(nef,z)
    index=len(nef)-2
    ar = (np.zeros(index)).tolist()

    for i in range(index):
        ar[i] = ((z[i]*(dndz[i+1]-dndz[i+2])+z[i+1]*(dndz[i+2]-dndz[i])
                  +z[i+2]*(dndz[i]-dndz[i+1])))/2
    return sum(ar)

"""
    FUNCIONES PARA RESOLVER LA ECUACIÓN DE BOLZMANN-MATANO
"""

def parabola(deltanef, z, index): # index = lista con tres enteros
    
        """
            Calcula parábola a partir de 3 puntos dados, útil para extrapolar 
        """
        y1, x1 = [z[index[0]],deltanef[index[0]]] 
        y2, x2 = [z[index[1]],deltanef[index[1]]]
        y3, x3 = [z[index[2]],deltanef[index[2]]]
        
        denom = (x1-x2) * (x1-x3) * (x2-x3)
        A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
        B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
        C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
        return A,B,C

def areaparabola (deltanef, z, index):
    
    """
        Calcula el área bajo una parábola obtenida con la funcion anterior
    """
    a, b, c = parabola(deltanef, z, index)
    x1, x2 = [0, deltanef[0]]
    return a/3.0 * ((x2)**3-(x1)**3) + b/2.0 * ((x2)**2-(x1)**2) +c*(x2-x1)

def integral(deltanef, z):
    
    """
        Hace la integral de z sobre deltanef como una suma acumulativa
        de trapecios con espaciado variable   
    """
    
    integrales = []
    integrales.append(0) # este 0 lo vamos a convertir en el area de la parábola
    
    ar = areaparabola(deltanef,z,[0,1,2])
    
    for i in range(len(z)-1):
        
        sumas = []
        s = 0
        
        for j in range(i+1):
            
            a = ((deltanef[j+1]-deltanef[j])*(z[j]+z[j+1])/2) # trapecios
            sumas.append(a)
            s = sum(sumas)
            
        integrales.append(s) # valor absoluto por ser un área
        
    return integrales + ar*np.ones(len(integrales)) # sumamos a todo el área de la parábola



