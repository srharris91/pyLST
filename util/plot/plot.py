import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms

def add_legend_outside(fig,ax, x0=1,y0=0.9, direction = "h", padpoints = 3,**kwargs):
    otrans = ax.figure.transFigure
    t = ax.legend(bbox_to_anchor=(x0,y0), loc=1, bbox_transform=otrans,**kwargs)
    plt.tight_layout()
    ax.figure.canvas.draw()
    plt.tight_layout()
    ppar = [0,-padpoints/72.] if direction == "v" else [-padpoints/72.,0] 
    trans2=matplotlib.transforms.ScaledTranslation(ppar[0],ppar[1],fig.dpi_scale_trans)+\
             ax.figure.transFigure.inverted() 
    tbox = t.get_window_extent().transformed(trans2 )
    bbox = ax.get_position()
    if direction=="v":
        ax.set_position([bbox.x0, bbox.y0,bbox.width, tbox.y0-bbox.y0]) 
    else:
        ax.set_position([bbox.x0, bbox.y0,tbox.x0-bbox.x0, bbox.height])

# plot S2 and SPE items together
def plot_SPE_and_DNS(S1,X1,Y1,S2,X2,Y2,label='',scale='linear',cmap=plt.cm.bwr,inline=False,title1='SPE',title2='DNS'):
    minimumSPE=S1.min()
    minimumDNS=S2.min()
    maximumSPE=S1.max()
    maximumDNS=S2.max()
    minimum = np.min([minimumSPE,minimumDNS])
    maximum = np.max([maximumSPE, maximumDNS])
    level = np.max([np.abs(minimum),np.abs(maximum)])
    if scale=='linear':
        levels=np.linspace(-level,level,256)
    else: 
        levels=np.linspace(minimum,maximum,256)
    
    fig,[ax1,ax2] = plt.subplots(ncols=2,sharex=True,sharey=True,figsize=(8,3))
    ax1.contourf(X1,Y1,S1,    levels,cmap=cmap)
    cs = ax2.contourf(X2,Y2,S2,    levels,cmap=cmap)
    ax1.set_xlabel('x')
    ax2.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(title1)
    ax2.set_title(title2)
    fig.tight_layout()
    cbar = fig.colorbar(cs,ax=[ax1,ax2])#,label=label)
    if inline:
        CS0 = ax1.contour(X1, Y1, S1,levels[::52],colors=('k',))
        ax1.clabel(CS0, inline=inline,colors='k')
        cbar.add_lines(CS0)
        CS1 = ax2.contour(X2, Y2, S2,levels[::52],colors=('k',))
        ax2.clabel(CS1, inline=inline,colors='k')
        cbar.add_lines(CS1)
    #cs.cmap.set_under('yellow')
    #cs.cmap.set_over('cyan')
    ax2.text(0.9, 0.9, label, horizontalalignment='right',
        verticalalignment='top', transform=ax2.transAxes)
    return fig,[ax1,ax2]

def plot_UVP(axs,U,V,P,y,yP,line='-',label=None):
    axU,axV,axP=axs
    axU.plot(U,y,line,label=label)
    axV.plot(V,y,line,label=label)
    axP.plot(P,yP,line,label=label)

def subplots(**kwargs):
    figsize = kwargs.pop('figsize',(4.75,4))
    tight_layout = kwargs.pop('tight_layout',True)
    dpi = kwargs.pop('dpi',200)
    fig,ax=plt.subplots(figsize=figsize,tight_layout=tight_layout,dpi=dpi,**kwargs)
    return fig,ax

def figure(**kwargs):
    figsize = kwargs.pop('figsize',(4.75,4))
    tight_layout = kwargs.pop('tight_layout',True)
    dpi = kwargs.pop('dpi',200)
    fig=plt.figure(figsize=figsize,tight_layout=tight_layout,dpi=dpi,**kwargs)
    return fig

def plot_eigen_sol(params,alphas,eigfuncrs,figsize=(4.75,6),fig=None,ax=None,color=None,marker='.',label=None):
    ny = params['grid'].ny
    y = params['grid'].y
    if fig==None:
        fig= figure(figsize=figsize)
        axa=plt.subplot(2,1,1)
    else:
        axa = ax[0]
    if np.all(color==None):
        axa.plot(alphas.real,alphas.imag,marker,picker=5,label=label)
    elif type(color)==str:
        axa.plot(alphas.real,alphas.imag,color+marker,picker=5,label=label)
    else:
        axa.scatter(alphas.real,alphas.imag,marker=marker,c=color,picker=5,label=label)
    axa.legend(loc='best',numpoints=1)
    if ax==None:
        axu=plt.subplot(2,4,5)
        axv=plt.subplot(2,4,6,sharey=axu)
        axw=plt.subplot(2,4,7,sharey=axu)
        axp=plt.subplot(2,4,8,sharey=axu)
    else:
        axu,axv,axw,axp = ax[1:]

    def format_coord(x,y):
        try:
            return r'α: {:g}'.format(x+y*1.j)
        except:
            return 'x: {}, y: {}'.format(x,y)
    axa.format_coord = format_coord

    def onpick(event):
        thisline = event.artist
        #print(event)
        try: # if plotted with plot
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            ind0 = ind[0]
            points = xdata[ind0],ydata[ind0]
            ax = event.artist.axes
            ax.set_title(r'$\alpha = {:g}+{:g}j$'.format(xdata[ind0],ydata[ind0]))
        except: # if scatter plot with color
            #thisline.set_offset_position('data')
            xy = thisline.get_offsets()
            array = thisline.get_array()
            xdata =xy[:,0]
            ydata =xy[:,1]
            ind = event.ind
            ind0 = ind[0]
            points = xdata[ind0],ydata[ind0]
            ax = event.artist.axes
            ax.set_title(r'$\alpha = {:g}+{:g}j, mag={:g}$'.format(xdata[ind0],ydata[ind0],array[ind0]))
        
        q = eigfuncrs[:4*ny,ind0]
        # normalize and extract components
        norm = q[np.argmax(np.abs(q[:3*ny]))]
        try:
            q /= norm
        except:
            pass
        q = q.reshape(-1,ny)
        u = q[0]
        v = q[1]
        w = q[2]
        p = q[3]
        #au = q[0]/(xdata[ind0]+ydata[ind0]*1.j)
        #av = q[1]/(xdata[ind0]+ydata[ind0]*1.j)
        #aw = q[2]/(xdata[ind0]+ydata[ind0]*1.j)
        #ap = q[3]/(xdata[ind0]+ydata[ind0]*1.j)
        for axi in [axu,axv,axw,axp]:
            axi.set_prop_cycle(None)
            if len(axi.lines)>0:
                for line in range(3):
                    axi.lines[0].remove()
        axu.plot(np.abs(u),y,'-',label='abs')
        axu.plot(u.real,y,'--',label='real')
        axu.plot(u.imag,y,'--',label='imag')
        axv.plot(np.abs(v),y,'-',label='abs')
        axv.plot(v.real,y,'--',label='real')
        axv.plot(v.imag,y,'--',label='imag')
        axw.plot(np.abs(w),y,'-',label='abs')
        axw.plot(w.real,y,'--',label='real')
        axw.plot(w.imag,y,'--',label='imag')
        axp.plot(np.abs(p),y,'-',label='abs')
        axp.plot(p.real,y,'--',label='real')
        axp.plot(p.imag,y,'--',label='imag')
        
        axu.legend(loc='best',numpoints=1);
        axu.relim()
        axv.relim()
        axw.relim()
        axp.relim()
        axu.autoscale_view()
        axv.autoscale_view()
        axw.autoscale_view()
        axp.autoscale_view()
        event.canvas.draw()
        
    axs=[axu,axv,axw,axp]
    axu.set_xlabel(r'$\hat{u}$')
    axv.set_xlabel(r'$\hat{v}$')
    axw.set_xlabel(r'$\hat{w}$')
    axp.set_xlabel(r'$\hat{p}$')
    for ax in axs:
        ax.set_ylabel(r'$y$')
        ax.label_outer()
        
    fig.canvas.mpl_connect('pick_event',onpick);
    return fig,[axa,axu,axv,axw,axp]

def plot_eigen_sol_4ntny(params,alphas,eigfuncrs,figsize=(4.75,6),fig=None,ax=None,color=None,marker='.'):
    ny = params['grid'].ny
    nt = params['grid'].nt
    y = params['grid'].y
    if fig==None:
        fig= figure(figsize=figsize)
        axa=plt.subplot(2,1,1)
    if np.all(color==None):
        axa.plot(alphas.real,alphas.imag,marker,picker=5)
    elif type(color)==str:
        axa.plot(alphas.real,alphas.imag,color+marker,picker=5)
    else:
        axa.scatter(alphas.real,alphas.imag,marker=marker,c=color,picker=5)
    if ax==None:
        axu=plt.subplot(2,4,5)
        axv=plt.subplot(2,4,6,sharey=axu)
        axw=plt.subplot(2,4,7,sharey=axu)
        axp=plt.subplot(2,4,8,sharey=axu)

    def format_coord(x,y):
        try:
            return r'α: {:g}'.format(x+y*1.j)
        except:
            return 'x: {}, y: {}'.format(x,y)
    axa.format_coord = format_coord

    def onpick(event):
        thisline = event.artist
        #print(event)
        try: # if plotted with plot
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            ind0 = ind[0]
            points = xdata[ind0],ydata[ind0]
            ax = event.artist.axes
            ax.set_title(r'$\alpha = {:g}+{:g}j$'.format(xdata[ind0],ydata[ind0]))
        except: # if scatter plot with color
            #thisline.set_offset_position('data')
            xy = thisline.get_offsets()
            array = thisline.get_array()
            xdata =xy[:,0]
            ydata =xy[:,1]
            ind = event.ind
            ind0 = ind[0]
            points = xdata[ind0],ydata[ind0]
            ax = event.artist.axes
            ax.set_title(r'$\alpha = {:g}+{:g}j, mag={:g}$'.format(xdata[ind0],ydata[ind0],array[ind0]))
        
        q = eigfuncrs[:,ind0]
        # normalize and extract components
        q = q.reshape(4,nt,ny)
        u = q[0].T # to be shape ny,nt
        v = q[1].T
        w = q[2].T
        p = q[3].T
        #au = q[0]/(xdata[ind0]+ydata[ind0]*1.j)
        #av = q[1]/(xdata[ind0]+ydata[ind0]*1.j)
        #aw = q[2]/(xdata[ind0]+ydata[ind0]*1.j)
        #ap = q[3]/(xdata[ind0]+ydata[ind0]*1.j)
        for axi in [axu,axv,axw,axp]:
            axi.set_prop_cycle(None)
            nlines = len(axi.lines)
            if nlines>0:
                for line in range(nlines):
                    axi.lines[0].remove()
        #axu.plot(np.abs(u[:,0]),y,'-',color='C0',label='abs')
        #axu.plot(u.real[:,0],y,'--',color='C1',label='real')
        #axu.plot(u.imag[:,0],y,'--',color='C2',label='imag')
        #axu.plot(np.abs(u[:,1:]),y,'-',color='C0')
        #axu.plot(u.real[:,1:],y,'--',color='C1')
        #axu.plot(u.imag[:,1:],y,'--',color='C2')
        axu.plot(np.abs(u),y,'-',color='C0')
        axu.plot(u.real,y,'--',color='C1')
        axu.plot(u.imag,y,':',color='C2')

        axv.plot(np.abs(v),y,'-',color='C0')
        axv.plot(v.real,y,'--',color='C1')
        axv.plot(v.imag,y,':',color='C2')

        axw.plot(np.abs(w),y,'-',color='C0')
        axw.plot(w.real,y,'--',color='C1')
        axw.plot(w.imag,y,':',color='C2')

        axp.plot(np.abs(p),y,'-',color='C0')
        axp.plot(p.real,y,'--',color='C1')
        axp.plot(p.imag,y,':',color='C2')
        
        #axu.legend(loc='best',numpoints=1);
        axu.relim()
        axv.relim()
        axw.relim()
        axp.relim()
        axu.autoscale_view()
        axv.autoscale_view()
        axw.autoscale_view()
        axp.autoscale_view()
        event.canvas.draw()
        
    axs=[axu,axv,axw,axp]
    axu.set_xlabel(r'$\hat{u}$')
    axv.set_xlabel(r'$\hat{v}$')
    axw.set_xlabel(r'$\hat{w}$')
    axp.set_xlabel(r'$\hat{p}$')
    for ax in axs:
        ax.set_ylabel(r'$y$')
        ax.label_outer()
        
    fig.canvas.mpl_connect('pick_event',onpick);
    return fig,[axa,axu,axv,axw,axp]

def plot_eigen_spectrum(alphas,color=None,fig=None,ax=None,figsize=(4.75,4),label=None,marker='.'):
    if (fig == None) and (ax==None):
        fig,ax= subplots(figsize=figsize)
    if np.all(color==None):
        ax.plot(alphas.real,alphas.imag,picker=5,linestyle='None',marker=marker,label=label)
    elif type(color)==str:
        ax.plot(alphas.real,alphas.imag,color=color,linestyle='None',marker=marker,picker=5,label=label)
    else:
        ax.scatter(alphas.real,alphas.imag,linestyle='None',c=color,marker=marker,picker=5,label=label)

    def format_coord(x,y):
        try:
            return r'α: {:g}'.format(x+y*1.j)
        except:
            return 'x: {}, y: {}'.format(x,y)
    ax.format_coord = format_coord

    def onpick(event):
        thisline = event.artist
        #print(event)
        try: # if plotted with plot
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            ind0 = ind[0]
            points = xdata[ind0],ydata[ind0]
            ax = event.artist.axes
            ax.set_title(r'$\alpha = {:g}+{:g}j$'.format(xdata[ind0],ydata[ind0]))
        except: # if scatter plot with color
            #thisline.set_offset_position('data')
            xy = thisline.get_offsets()
            array = thisline.get_array()
            xdata =xy[:,0]
            ydata =xy[:,1]
            ind = event.ind
            ind0 = ind[0]
            points = xdata[ind0],ydata[ind0]
            ax = event.artist.axes
            ax.set_title(r'$\alpha = {:g}+{:g}j, mag={:g}$'.format(xdata[ind0],ydata[ind0],array[ind0]))
        
        event.canvas.draw()
        
    ax.set_xlabel(r'$\alpha_r$')
    ax.set_ylabel(r'$\alpha_i$')
        
    fig.canvas.mpl_connect('pick_event',onpick);
    return fig,ax

def plot_state_vector(params,q,fig=None,ax=None,color='C0',figsize=(6.75,4),label=None,marker=None,linestyle='-'):
    y = params['grid'].y
    ny = params['grid'].ny
    nt = params['grid'].nt
    u,v,w,p = q.reshape(4,nt,ny)
    if (fig == None) and (ax==None):
        fig,ax= subplots(figsize=figsize,ncols=4,sharey=True)
    axu,axv,axw,axp = ax
    # transpose such that y is first dimension
    u = u.T
    v = v.T
    w = w.T
    p = p.T

    axu.plot(u,y,color=color,marker=marker,linestyle=linestyle)
    axv.plot(v,y,color=color,marker=marker,linestyle=linestyle)
    axw.plot(w,y,color=color,marker=marker,linestyle=linestyle)
    axp.plot(p,y,color=color,marker=marker,linestyle=linestyle)
    #axu.plot(np.abs(u),y,'C0',marker=marker)
    #axu.plot(u.real,y,'C1--',marker=marker)
    #axu.plot(u.imag,y,'C2--',marker=marker)

    #axv.plot(np.abs(v),y,'C0',marker=marker)
    #axv.plot(v.real,y,'C1--',marker=marker)
    #axv.plot(v.imag,y,'C2--',marker=marker)

    #axw.plot(np.abs(w),y,'C0',marker=marker)
    #axw.plot(w.real,y,'C1--',marker=marker)
    #axw.plot(w.imag,y,'C2--',marker=marker)


    axu.set_xlabel(r'$u$')
    axv.set_xlabel(r'$v$')
    axw.set_xlabel(r'$w$')
    axp.set_xlabel(r'$p$')
    axu.set_ylabel(r'$y$')
        
    return fig,ax

def plot_eigen_vector(params,q,fig=None,ax=None,color='C0',figsize=(6.75,4),label=None,marker=None,linestyle='-'):
    y = params['grid'].y
    ny = params['grid'].ny
    nt = params['grid'].nt
    u,v,w,p = q.reshape(4,ny)
    if (fig == None) and (ax==None):
        fig,ax= subplots(figsize=figsize,ncols=4,sharey=True)
    axu,axv,axw,axp = ax
    # transpose such that y is first dimension
    u = u
    v = v
    w = w
    p = p

    axu.plot(u,y,color=color,marker=marker,linestyle=linestyle)
    axv.plot(v,y,color=color,marker=marker,linestyle=linestyle)
    axw.plot(w,y,color=color,marker=marker,linestyle=linestyle)
    axp.plot(p,y,color=color,marker=marker,linestyle=linestyle)

    axu.set_xlabel(r'$\hat{u}$')
    axv.set_xlabel(r'$\hat{v}$')
    axw.set_xlabel(r'$\hat{w}$')
    axp.set_xlabel(r'$\hat{p}$')
    axu.set_ylabel(r'$y$')
        
    return fig,ax

def plot_baseflow(params,baseflow,fig=None,ax=None,color='C0',figsize=(6.75,4),label=None,marker=None,linestyle='-'):
    y = params['grid'].y
    ny = params['grid'].ny
    u,v,w,p = q.reshape(4,ny)
    U = baseflow.get_U()
    V = baseflow.get_V()
    P = baseflow.get_P()
    if np.any(P==None):
        P = np.zeros_like(U)
    if (fig == None) and (ax==None):
        fig,ax= subplots(figsize=figsize,ncols=3,sharey=True)
    axu,axv,axp = ax

    axu.plot(U,y,color=color,marker=marker,linestyle=linestyle)
    axv.plot(V,y,color=color,marker=marker,linestyle=linestyle)
    axp.plot(P,y,color=color,marker=marker,linestyle=linestyle)

    axu.set_xlabel(r'$U$')
    axv.set_xlabel(r'$V$')
    axp.set_xlabel(r'$P$')
    axu.set_ylabel(r'$y$')
        
    return fig,ax

