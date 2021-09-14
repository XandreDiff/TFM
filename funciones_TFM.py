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
    
    n_0 = índice de superficie de la guía óptica, tipo float
    n_m = índices efectivos medido, tipo list
    n_c = índice de la cubierta de la guía típicamente aire =1, tipo float
    pol = selecciona tipo de polarización de la luz, tipo str
    lmda = longitud de onda del láser empleado, tipo float
    
"""

def wh_pol(n_0, n_m, n_c, pol, lmda): 
    
      """ 
      
        Reconstruye los puntos de retorno z suponiendo que el perfil n(x)
        se construye linealmente a partir de los indices efectivos n_m[i]
        y teniendo en cuenta luz TE o TM
        
    """
        
    nef = (np.zeros(len(n_m)+1)).tolist()
    nef[0] = np.float64(n_0)
    nef[1:len(nef)] = n_m[:]
    
    """
    nef = variable auxiliar para colocar n_0  y n_m en la misma lista
    
    """
    
    if pol = "TM":
        a_m = (nef[0]/n_c)**4  # factor de asimetría TM
    elif pol = "TE":
        a_m = 1
    else :
        print("Polarización mal introducida, debe ser "TE" o "TM" ")
     
    """
    
    Se escoge la polarización TE o TM, si se selecciona mal devuelve error
    
    """
    
    z = (np.zeros(len(nef))).tolist() # vector de zeros de indice m=0,1,2...M
    z[0] = np.float64(0) 
    z[1] = lmda*((3.0/2.0)*((nef[0]+3.0*nef[1])/2.0)**(-0.5)*(nef[0]-nef[1])**(-0.5)*(0.125 
           + (1/(2.0*np.pi))*np.arctan(a_m*(((nef[1])**2-nc**2)/((nef[0])**2-(nef[1])**2))**(0.5))))
    
    for m in range(2, len(nef)): # desde m = 2
    
        s = 0
        sumas = [] # para poder hacer el sumatorio hasta M-1, vector de sumandos
        
        for j in range(1, m):  
            
            a = (((nef[j-1]+nef[j])/2.0 + nef[m])**(0.5)*((z[j]-z[j-1])/(nef[j-1]-nef[j]))
                 *((nef[j-1]-nef[m])**(1.5)-(nef[j]-nef[m])**(1.5)))
            sumas.append(a)
            
        s = sum(sumas)  # suma de los sumandos para cada j  
        
        z[m] = lmda*(z[m-1] + ((3.0/2.0)*((nef[m-1]+3.0*nef[m])/2.0)**(-0.5)*(nef[m-1]-nef[m])**(-0.5))
                *((4.0*m -3)/8 + (1/(2.0*np.pi))
                  *np.arctan(a_m*(((nef[m])**2-nc**2)/((nef[0])**2-(nef[m])**2))**(0.5))
                  -(2.0/3.0)*s))
        """
        
        Se obtienen los puntos de retorno segun WH modificado en micras
        
        """
        
    return z, nef

      
'''
    FUNCIÓN PARA MINIMIZAR n_0:  AREAS DE TRIÁNGULOS CON DERIVADAS
'''

def minimozip(l1,l2):
    
    """ 
        Para cada par de puntos (funcion, n0) selecciona la pareja que haga
        funcion minimal
    """
    
    zippedlist = list(zip(l1,l2))
    return sorted(zippedlist, reverse = False, key = lambda x:x[1])[0]

def derh(x, y):
    
    """
        Hace la derivada con espaciado constante dy/dx =  dy/dm * dm/dx con m=1
    """
    
    dx = (np.zeros(len(x))).tolist()
    
    dx[0] = 0.5*(-3.0*x[0]+4.0*x[1]-x[2]) # esquema a dos vecinos para delante en el extremo
    dx[len(x)-1] = 0.5*(x[len(x)-3]-4.0*x[len(x)-2]+3.0*x[len(x)-1]) # idem a vecinos para atrás
    
    for i in range(1, len(x)-1):
        dx[i] = 0.5*(x[i+1]-x[i-1]) # esquema de diferencias centrales
        
    """
    Término dx/dm
    
    """
    
    dy = (np.zeros(len(y))).tolist()
    
    dy[0] = 0.5*(-3.0*y[0]+4.0*y[1]-y[2])
    dy[len(y)-1] = (0.5*(y[len(y)-3]
                   -4.0*y[len(y)-2]+3.0*y[len(y)-1]))
    
    for i in range(1, len(y)-1):
        dy[i] = 0.5*(y[i+1]-y[i-1])

    """
    Término dy/dm
    
    """
    dydx= [dy[i]/dx[i] for i in range(len(y))]
    return dydx

def area_rogozinski(n_0, n_m):
    
    """
        Calculo de la suma de las areas de los tríangulos cuyos vértices vienen
        dados por las coordenadas (z[i], dn[i]/dz[i]) 
        
    """
    
    z, nef = wh_pol(n_0, n_m)
    dndz = derh(z, nef)
    index=len(nef)-2
    ar = (np.zeros(index)).tolist()

    for i in range(index):
        ar[i] = ((z[i]*(dndz[i+1]-dndz[i+2])+z[i+1]*(dndz[i+2]-dndz[i])
                  +z[i+2]*(dndz[i]-dndz[i+1])))/2
    return sum(ar)

"""
    FUNCIONES PARA RESOLVER LA ECUACIÓN DE BOLTZMANN-MATANO
    
    deltanef = parámetro proporcional a la concentración dado por ec de Lorentz-Lorenz
                puede ser en aproximación lineal o sin ella
"""

def parabola(deltanef, z, index): 
    
        """
            Calcula parábola a partir de 3 puntos dados, útil para extrapolar la cola asintótica
            para concentraciones pequeñas, sustituye el area bajo la curva tipo erfc(x)
            
            index = lista con tres enteros, tipo list
        """
        y1, x1 = [z[index[0]],deltanef[index[0]]] 
        y2, x2 = [z[index[1]],deltanef[index[1]]]
        y3, x3 = [z[index[2]],deltanef[index[2]]]
        
        denom = (x1-x2) * (x1-x3) * (x2-x3)
        A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
        B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
        C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
        return A, B, C

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
    integrales.append(0) # este 0 lo vamos a convertir en el area bajo la parábola
    
    parabol = areaparabola(deltanef, z, [0, 1, 2])
    
    for i in range(len(z)-1):
        
        sumas = []
        s = 0
        
        for j in range(i+1):
            
            a = ((deltanef[j+1]-deltanef[j])*(z[j]+z[j+1])/2) 
            sumas.append(a)
            s = sum(sumas)
            
        integrales.append(s) 
        
        """
        Trapecios de altura el valor medio z[j]+z[j+1])/2 y base la diferencia deltanef[j+1]-deltanef[j]
        A continuación se suma adicionalmente el pequeño valor de la parábola
        """
        
    return integrales + parabol*np.ones(len(integrales)) 



