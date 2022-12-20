import numpy as np
from scipy.integrate import cumtrapz
#from ..Classes import base_flow_class
#from ..helper import ifflag
from ..base_flow import channel
from ..base_flow import blasius
from ..Classes import grid1D_omega_beta_class
from ..Classes import base_flow_class
from ..grid import set_D
from ..grid import set_Cheby_mapped_y
from ..LST import dLdomega
from ..LST import LST
from ..LST import LST_alphas
from ..PSE import march_PSE2D_multistep
import tkinter as tk
import tkinter.ttk
import tkinter.messagebox
import tkinter.filedialog
import ttkthemes
import matplotlib.collections
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
#from matplotlib.figure import Figure

import numpy as np
import h5py

plt.style.use('seaborn-notebook')

class MyWindow:
    def __init__(self, win):
        self.win=win
        self.p1 = tk.PhotoImage(file='/home/shaun/GITHUBs/SPE/pySPE/util/GUI/spe2_black.png')
        self.win.wm_iconphoto(True, self.p1)
        self.win.title("SPE GUI")
        #self.win.wm_iconbitmap(r'/home/shaun/Desktop/Shaun_tkinter/py.ico')
        #p1 = tk.PhotoImage(file='py.png')
        #self.win.iconphoto(False,p1)
        #self.win.wm_title("Plot in Tk")
        # add a couple menu items
        self.menu = tk.Menu(self.win)
        self.File_items = tk.Menu(self.menu,tearoff=False)
        self.File_items.add_command(label='New',command=self._new,accelerator='Ctrl+n')
        self.File_items.add_command(label='Open',command=self._open,accelerator='Ctrl+o')
        self.File_items.add_command(label='Save',command=self._save,accelerator='Ctrl+s')
        self.File_items.add_command(label='Save as',command=self._saveas,accelerator='Ctrl+Shift+S')
        self.File_items.add_command(label='Close',command=self._quit,accelerator='Esc')
        self.menu.add_cascade(label='File',menu=self.File_items)
        self.win.config(menu=self.menu)
        # keybindings
        self.win.bind('<Control-o>', lambda event:self._open())
        self.win.bind('<Control-s>', lambda event:self._save())
        self.win.bind('<Control-n>', lambda event:self._new())
        self.win.bind('<Control-Shift-S>', lambda event:self._saveas())
        self.win.bind('<Escape>', lambda event:self._quit())
        # add some tab controls
        self.tab_control = tk.ttk.Notebook(self.win)
        self.tab1 = tk.ttk.Frame(self.tab_control)
        self.tab2 = tk.ttk.Frame(self.tab_control)
        #self.tab3 = tk.ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text='LST')
        self.tab_control.add(self.tab2, text='LPSE')
        #self.tab_control.add(self.tab3, text='NPSE')
        #self.lbl3 = tk.ttk.Label(self.tab3, text= 'March the eigenfunction/eigenvalue pair using the non-linear Parabolized Stability Equations (NPSE)')
        #self.lbl3.pack()
        self.tab_control.pack(expand=1, fill='both')
        # notebook when change tabs
        #self.win.bind("<<NotebookTabChanged>>", self._refresh_pse_params)
        
        # LST items
        # add radio buttons
        self.fLST1 = tk.ttk.Frame(self.tab1)
        self.lbl1 = tk.ttk.Label(self.fLST1, text= 'Use local Linear Stability Theory (LST) to save\n the desired eigenfunction/eigevalue pair')#,padx=25,pady=25)
        self.lbl1.grid(row=0,column=0,columnspan=3)
        #lomega = tk.ttk.Label(self.f1,text=u'\u03C9 or \u03B1')
        lomega = tk.ttk.Label(self.fLST1,text=u'\u03C9'); lomega.grid(  row=1,column=0)
        self.omega = tk.ttk.Entry(self.fLST1,width=10); self.omega.grid(row=1,column=1)
        lbeta = tk.ttk.Label(self.fLST1,text=u'\u03B2'); lbeta.grid(    row=2,column=0)
        self.beta = tk.ttk.Entry(self.fLST1,width=10); self.beta.grid(  row=2,column=1)
        lRe = tk.ttk.Label(self.fLST1,text='Re'); lRe.grid(             row=3,column=0)
        self.Re = tk.ttk.Entry(self.fLST1,width=10); self.Re.grid(      row=3,column=1)
        lny = tk.ttk.Label(self.fLST1,text='ny'); lny.grid(             row=4,column=0)
        self.ny = tk.ttk.Entry(self.fLST1,width=10); self.ny.grid(      row=4,column=1)
        ly = tk.ttk.Label(self.fLST1,text='y range'); ly.grid(          row=5,column=0)
        self.y0 = tk.ttk.Entry(self.fLST1,width=10); self.y0.grid(      row=5,column=1)
        self.y1 = tk.ttk.Entry(self.fLST1,width=10); self.y1.grid(      row=5,column=2)
        # dropdown menus
        #ldimension = tk.ttk.Label(self.f1,text='Dimensions'); ldimension.grid(                  row=5,column=0)
        #leigenvalue = tk.ttk.Label(self.f1,text='Eigenvalue'); leigenvalue.grid(                row=5,column=1)
        #self.dropdown1var,self.dropdown1 = self.set_dropdown(self.f1,['1D','2D','3D'],          row=6,column=0)
        #self.dropdown2var,self.dropdown2 = self.set_dropdown(self.f1,[u'\u03B1',u'\u03C9'],     row=6,column=1)
        leigenvalue = tk.ttk.Label(self.fLST1,text='Spectrum Soln.'); leigenvalue.grid(            row=8,column=0)
        #leigenvalue_num = tk.ttk.Label(self.f1,text='# of modes'); leigenvalue_num.grid(            row=7,column=1)
        leigenvalue_guess = tk.ttk.Label(self.fLST1,text='Guess'); leigenvalue_guess.grid(               row=8,column=1)
        self.full_partial_var,self.full_partial_dim = self.set_dropdown(self.fLST1,['full','single'],row=9,column=0)
        #self.eigenvalue_num = tk.ttk.Entry(self.fLST1,width=10); self.eigenvalue_num.grid(     row=8,column=1)
        self.eigenvalue_guess = tk.ttk.Entry(self.fLST1,width=10); self.eigenvalue_guess.grid(     row=9,column=1)
        #self.id_on_spectrum = tk.ttk.Checkbutton(self.fLST1,text='plot id')
        #self.id_on_spectrum.grid(                                                               row=9,column=0)
        # solve, plot, save buttons
        self.button1_solve = tk.ttk.Button(master=self.fLST1, text="Solve", command=self._solve)
        self.button1_solve.grid(row=11,column=0)
        self.button2_plot = tk.ttk.Button(master=self.fLST1, text="Plot", command=self._plot_spectrum)
        self.button2_plot.grid(row=11,column=1)
        self.button3_save = tk.ttk.Button(master=self.fLST1, text="Save", command=self._save_spectrum)#, command=self._solve)
        self.button3_save.grid(row=12,column=0)
        # ID and eigenfunction plots
        lid = tk.ttk.Label(self.fLST1,text='id'); lid.grid(        row=13,column=0)
        self.lid = tk.ttk.Label(self.fLST1,text=''); self.lid.grid(row=13,column=1)
        # add plot and save to eigenfunction
        self.button4_plot = tk.ttk.Button(master=self.fLST1, text="Plot id", command=self._plot_id)
        self.button4_plot.grid(row=14,column=0)
        self.button5_save = tk.ttk.Button(master=self.fLST1, text="Save id", command=self._save_id)
        self.button5_save.grid(row=14,column=1)
        # add eigenvalue spot
        #lalpha = tk.ttk.Label(self.f1,text=u'\u03C9 or \u03B1 of id'); lalpha.grid(row=14,column=0)
        lalpha = tk.ttk.Label(self.fLST1,text=u'\u03B1 of id'); lalpha.grid(row=15,column=0)
        self.lalpha = tk.ttk.Label(self.fLST1,text=''); self.lalpha.grid(row=15,column=1)
        #self.rad1 = tk.ttk.Radiobutton(self.f1,text='sin',value=1,command=self.plot_sin,)
        #self.rad1.grid(column=0,row=16)
        #self.rad2 = tk.ttk.Radiobutton(self.f1,text='cos',value=2,command=self.plot_cos)
        #self.rad2.grid(column=0,row=16)
        # add text latex to tab2
        #label1 = tk.ttk.Label(self.f1)
        #label1.grid(row=6,column=0)
        #self.latex1 = self.text_graph(r'$x$',label1,row=6,column=0)
        #self.fLST1.pack(side=tk.LEFT)
        self.fLST1.grid(row=0,column=0,rowspan=2)
        # plot figure
        self.fig_eigenvalues = plt.Figure(figsize=(7.3, 3.5),tight_layout=True)#, dpi=100)
        self.t = np.arange(0, 3, .01)
        self.ax_eigenvalues = self.fig_eigenvalues.add_subplot(111)
        self.eigenvalues_line, = self.ax_eigenvalues.plot(self.t,self.t,'o',picker=1)
            #self.t, 
            #2 * np.sin(2 * np.pi * self.t),label=r'$2\sin{(2 \pi t)}$')
        #self.rad1.invoke()
        #self.ax_eigenvalues.legend(loc='best',numpoints=1)
        self.ax_eigenvalues.set_xlabel(r'$\alpha_r$')
        self.ax_eigenvalues.set_ylabel(r'$\alpha_i$')
        #self.fig_eigenvalues.tight_layout()
        self.annotate = self.ax_eigenvalues.annotate('',xy=self.t[:2],xytext=(0.5,1.01),textcoords='axes fraction')#,arrowprops=dict(facecolor='black',arrowstyle="-"))
        #self.ax_eigenvalues.set_picker(300.)
        #self.plot_sin()
        # export figure to canvas_eigenvalues widget
        self.fLST2 = tk.ttk.Frame(self.tab1)
        #self.fLST2.pack(fill=tk.BOTH, expand=1)#side=tk.LEFT)
        self.fLST2.grid(row=0,column=1)
        self.canvas_eigenvalues = FigureCanvasTkAgg(self.fig_eigenvalues, master=self.fLST2)  # A tk.DrawingArea.
        #self.canvas_eigenvalues.draw()
        #self.canvas_eigenvalues.get_tk_widget().pack()#side=tk.BOTTOM)#, fill=tk.BOTH, expand=1)
        # get NavigationToolbar widget
        #self.toolbar_frame_eigenvalues = tk.ttk.Frame(self.tab1)
        #self.toolbar_frame_eigenvalues = self.f2
        self.toolbar_eigenvalues = NavigationToolbar2Tk(self.canvas_eigenvalues, self.fLST2)
        #self.toolbar_eigenvalues.configure(background='#efebe7')
        #print(self.toolbar_eigenvalues.keys())
        #self.toolbar_eigenvalues.configure(fg='#efebe7')
        self.toolbar_eigenvalues.update()
        self.canvas_eigenvalues.get_tk_widget().pack()#side=tk.BOTTOM)#, fill=tk.BOTH, expand=1)
        # trace key presses
        #self.canvas_eigenvalues.mpl_connect("key_press_event", self.on_key_press_eigenvalues)
        # trace picking on figure
        self.annot = self.ax_eigenvalues.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),arrowprops=dict(arrowstyle="-"))
        self.annot.set_visible(False)
        self.canvas_eigenvalues.mpl_connect('motion_notify_event',self.hover)
        self.canvas_eigenvalues.mpl_connect('pick_event', self.onpick)
        # have annotation in top corner
        self.annotate = self.ax_eigenvalues.text(0.5,1.01,'index={: 4d}, data={:.2e},{:.2e}'.format(-1,0.,0.),transform=self.ax_eigenvalues.transAxes)
        # add eigenfunction figure
        self.fig_eigenfunction = plt.Figure(figsize=(7.3, 3.5),tight_layout=True)#, dpi=100)
        self.ax_ubar = self.fig_eigenfunction.add_subplot(151)
        self.ax_uhat = self.fig_eigenfunction.add_subplot(152,sharey=self.ax_ubar)
        self.ax_vhat = self.fig_eigenfunction.add_subplot(153,sharey=self.ax_ubar)
        self.ax_what = self.fig_eigenfunction.add_subplot(154,sharey=self.ax_ubar)
        self.ax_phat = self.fig_eigenfunction.add_subplot(155,sharey=self.ax_ubar)
        self.ax_uhat.tick_params(labelleft=False)
        self.ax_vhat.tick_params(labelleft=False)
        self.ax_what.tick_params(labelleft=False)
        self.ax_phat.tick_params(labelleft=False)
        self.ax_ubar.set_ylabel(r'$y$')
        self.ax_ubar.set_xlabel(r'$\overline{u}$')
        self.ax_uhat.set_xlabel(r'$\hat{u}$')
        self.ax_vhat.set_xlabel(r'$\hat{v}$')
        self.ax_what.set_xlabel(r'$\hat{w}$')
        self.ax_phat.set_xlabel(r'$\hat{p}$')
        self.ubar_line, = self.ax_ubar.plot(self.t,self.t,'-') # plot zeros on ubar
        self.uhatline1, = self.ax_uhat.plot(self.t,self.t,'-',label='real')
        self.uhatline2, = self.ax_uhat.plot(self.t,self.t,'--',label='imag')
        self.uhatline3, = self.ax_uhat.plot(self.t,self.t,'k-',label='abs')
        self.vhatline1, = self.ax_vhat.plot(self.t,self.t,'-',label='real')
        self.vhatline2, = self.ax_vhat.plot(self.t,self.t,'--',label='imag')
        self.vhatline3, = self.ax_vhat.plot(self.t,self.t,'k-',label='abs')
        self.whatline1, = self.ax_what.plot(self.t,self.t,'-',label='real')
        self.whatline2, = self.ax_what.plot(self.t,self.t,'--',label='imag')
        self.whatline3, = self.ax_what.plot(self.t,self.t,'k-',label='abs')
        self.phatline1, = self.ax_phat.plot(self.t,self.t,'-',label='real')
        self.phatline2, = self.ax_phat.plot(self.t,self.t,'--',label='imag')
        self.phatline3, = self.ax_phat.plot(self.t,self.t,'k-',label='abs')
        self.ax_uhat.legend(loc='best',numpoints=1)
        self.fLST3 = tk.ttk.Frame(self.tab1)
        #self.fLST3.pack(fill=tk.BOTH, expand=1)#side=tk.LEFT)
        self.fLST3.grid(row=1,column=1)
        self.canvas_eigenfunction = FigureCanvasTkAgg(self.fig_eigenfunction, master=self.fLST3)  # A tk.DrawingArea.
        #self.canvas_eigenfunction.draw()
        #self.canvas_eigenfunction.get_tk_widget().pack()#side=tk.BOTTOM)#, fill=tk.BOTH, expand=1)
        # get NavigationToolbar widget
        #self.toolbar_frame_eigenfunction = tk.ttk.Frame(self.tab1)
        #self.fLST2.pack(fill=tk.BOTH, expand=1)#side=tk.LEFT)
        self.toolbar_eigenfunction = NavigationToolbar2Tk(self.canvas_eigenfunction, self.fLST3)
        self.toolbar_eigenfunction.update()
        self.canvas_eigenfunction.get_tk_widget().pack()#side=tk.BOTTOM)#, fill=tk.BOTH, expand=1)
        #self.canvas_eigenfunction.mpl_connect("key_press_event", self.on_key_press_eigenfunction)

        # LPSE
        self.fPSE1 = tk.ttk.Frame(self.tab2)
        self.lbl2 = tk.ttk.Label(self.fPSE1, text='March the eigenfunction/eigenvalue pair using the\n linear Parabolized Stability Equations (LPSE)')
        self.lbl2.grid(row=0,column=0,columnspan=3)
        # grid params
        lomega_PSE = tk.ttk.Label(self.fPSE1,text=u'\u03C9'); lomega_PSE.grid(                      row=1,column=0)
        self.omega_PSE = tk.ttk.Label(self.fPSE1,text=self.omega.get()); self.omega_PSE.grid(       row=1,column=1)
        lbeta_PSE = tk.ttk.Label(self.fPSE1,text=u'\u03B2'); lbeta_PSE.grid(                        row=2,column=0)
        self.beta_PSE = tk.ttk.Label(self.fPSE1,text=self.beta.get()); self.beta_PSE.grid(          row=2,column=1)
        lRe = tk.ttk.Label(self.fPSE1,text='Re'); lRe.grid(                                         row=3,column=0)
        self.Re_PSE = tk.ttk.Label(self.fPSE1,text=self.Re.get()); self.Re_PSE.grid(                row=3,column=1)
        lny = tk.ttk.Label(self.fPSE1,text='ny'); lny.grid(                                         row=4,column=0)
        self.ny_PSE = tk.ttk.Label(self.fPSE1,text=self.ny.get()); self.ny_PSE.grid(                row=4,column=1)
        ly = tk.ttk.Label(self.fPSE1,text='y range'); ly.grid(                                      row=5,column=0)
        self.y0_PSE = tk.ttk.Label(self.fPSE1,text=self.y0.get()); self.y0_PSE.grid(                row=5,column=1)
        self.y1_PSE = tk.ttk.Label(self.fPSE1,text=self.y1.get()); self.y1_PSE.grid(                row=5,column=2)
        #self.fPSE1.grid(row=0,column=0)
        # add plots
        self.fig_PSE = plt.Figure(figsize=(7.3, 7.0),tight_layout=True)#, dpi=100)
        self.ax_PSEurms = self.fig_PSE.add_subplot(311)
        self.ax_PSEalphar = self.fig_PSE.add_subplot(312,sharex=self.ax_PSEurms)
        self.ax_PSEalphai = self.fig_PSE.add_subplot(313,sharex=self.ax_PSEurms)
        #self.ax_PSEu = self.fig_PSE.add_subplot(414,sharex=self.ax_PSEurms)
        self.ax_PSEurms.tick_params(labelbottom=False)
        self.ax_PSEalphar.tick_params(labelbottom=False)
        self.ax_PSEalphai.tick_params(labelbottom=False)
        self.ax_PSEurms.set_ylabel(r'$\max (u_\mathrm{rms})$')
        self.ax_PSEalphar.set_ylabel(r'$\alpha_r$')
        self.ax_PSEalphai.set_ylabel(r'$\alpha_i$')
        self.ax_PSEalphai.set_xlabel(r'$x$')
        #self.ax_PSEu.set_xlabel(r'$x$')
        #self.ax_PSEu.set_ylabel(r'$y$')
        self.PSEurms_line, = self.ax_PSEurms.plot(self.t,self.t,'-') # plot zeros on ubar
        self.PSEalphar_line, = self.ax_PSEalphar.plot(self.t,self.t,'-') # plot zeros on ubar
        self.PSEalphai_line, = self.ax_PSEalphai.plot(self.t,self.t,'-') # plot zeros on ubar
        #self.PSEu_line, = self.ax_PSEu.plot(self.t,self.t,'-') # plot zeros on ubar
        self.fPSE2 = tk.ttk.Frame(self.tab2)
        self.fPSE2.grid(row=0,column=1,rowspan=2)
        self.canvas_PSE = FigureCanvasTkAgg(self.fig_PSE, master=self.fPSE2)  # A tk.DrawingArea.
        self.toolbar_PSE = NavigationToolbar2Tk(self.canvas_PSE, self.fPSE2)
        self.toolbar_PSE.update()
        self.canvas_PSE.get_tk_widget().pack()#side=tk.BOTTOM)#, fill=tk.BOTH, expand=1)
        #self.canvas_PSE.mpl_connect("key_press_event", self.on_key_press_PSE)
        # add marching items for PSE
        #self.fPSE3 = tk.ttk.Frame(self.tab2)
        lalpha = tk.ttk.Label(self.fPSE1,text=u'\u03B1 of id'); lalpha.grid(                                                                        row=6,column=0)
        self.lalpha_PSE = tk.ttk.Label(self.fPSE1,text=self.lalpha['text']); self.lalpha_PSE.grid(                                                  row=6,column=1)
        self.button_refresh_pse = tk.ttk.Button(master=self.fPSE1, text="Get parameters and set hx", command=self._refresh_pse_params); self.button_refresh_pse.grid(  row=7,column=0,columnspan=3)
        lhx = tk.ttk.Label(self.fPSE1,text='hx'); lhx.grid(                                                                                         row=8,column=0)
        self.hx = tk.ttk.Entry(self.fPSE1,width=10); self.hx.grid(                                                                                  row=8,column=1)
        lsteps = tk.ttk.Label(self.fPSE1,text='steps'); lsteps.grid(                                                                                row=9,column=0)
        self.steps = tk.ttk.Entry(self.fPSE1,width=10); self.steps.grid(                                                                            row=9,column=1)
        self.button_march_pse = tk.ttk.Button(master=self.fPSE1, text="March", command=self._march_pse); self.button_march_pse.grid(                row=10,column=0)
        self.button_plot_pse = tk.ttk.Button(master=self.fPSE1, text="Plot", command=self._plot_pse); self.button_plot_pse.grid(                    row=10,column=1)
        self.button_save_pse = tk.ttk.Button(master=self.fPSE1, text="Save", command=self._save_pse); self.button_save_pse.grid(                    row=10,column=2)
        self.fPSE1.grid(row=0,column=0)

        # global buttons and file
        # file
        self.file=None
        self.file_h5py=None
        # add open button
        self.button1_open = tk.ttk.Button(master=self.win, text="Open", command=self._open)
        self.button1_open.pack(side=tk.LEFT,expand=True)
        # add Save button
        self.button2_save = tk.ttk.Button(master=self.win, text="Save", command=self._save)
        self.button2_save.pack(side=tk.LEFT,expand=True)
        # add quit button
        self.button3_quit = tk.ttk.Button(master=self.win, text="Quit", command=self._quit)
        self.button3_quit.pack(side=tk.LEFT,expand=True)
    def set_text(self,entry,text):
        entry.delete(0,tk.END)
        entry.insert(0,text)
        return
    def _march_pse(self):
        # normalize to 0.0025 (TODO could add norm factor entry)
        ny = self.params['ny']
        u_old1 = self.qhat0[:ny]
        norm1 = np.max(np.sqrt(2.*u_old1*u_old1.conj()))
        self.qhat0 = 0.0025*self.qhat0/norm1
        # add pse params
        self.params['alpha']=self.alpha0
        self.params['hx'] = float(self.hx.get())
        if self.params['hx']<1./self.alpha0.real:
            self.params['flags']['neglect_dPdx_term']=True
        else:
            self.params['flags']['neglect_dPdx_term']=False
        self.params['steps'] = int(self.steps.get())
        self.params['flags']['alpha_update']=True
        # march
        if self.params['x'] != self.params['x_start']:
            self.params['x'] = self.params['x_start']
            self.baseflow = blasius(self.params['grid'].y,x=self.params['x'],nu=self.params['nu'],plot=False)

        self.alpha_PSE,self.x_PSE,self.q_PSE = march_PSE2D_multistep(self.params,self.diffs,self.baseflow,self.qhat0,self.helper_mats)
    def _plot_pse(self):
        # postprocess
        ny = self.params['ny']
        self.Ialpha_PSE = cumtrapz(self.alpha_PSE,self.x_PSE-self.x_PSE[0],initial=0)
        uPSE = self.q_PSE[:,:ny] * np.exp(1.j*self.Ialpha_PSE[:,np.newaxis])
        self.uPSErms = np.max(np.sqrt(2.*uPSE*uPSE.conj()),axis=1)
        # update urms plot
        self.PSEurms_line.set_xdata(self.x_PSE)
        self.PSEurms_line.set_ydata(self.uPSErms.real)
        self.ax_PSEurms.relim() # recompute the ax_eigenvalues.dataLim
        self.ax_PSEurms.autoscale_view() # update ax_eigenvalues.viewLim using the new dataLim
        #self.ubar_line, = self.ax_ubar.plot(self.baseflow.get_U(),self.params['grid'].y)
        # update alphar
        self.PSEalphar_line.set_xdata(self.x_PSE)
        self.PSEalphar_line.set_ydata(self.alpha_PSE.real)
        self.ax_PSEalphar.relim() # recompute the ax_eigenvalues.dataLim
        self.ax_PSEalphar.autoscale_view() # update ax_eigenvalues.viewLim using the new dataLim
        # update alphai
        self.PSEalphai_line.set_xdata(self.x_PSE)
        self.PSEalphai_line.set_ydata(self.alpha_PSE.imag)
        self.ax_PSEalphai.relim() # recompute the ax_eigenvalues.dataLim
        self.ax_PSEalphai.autoscale_view() # update ax_eigenvalues.viewLim using the new dataLim
        # update contour
        self.canvas_PSE.mpl_connect("key_press_event", self.on_key_press_PSE)
        self.fig_PSE.canvas.draw()

    def _save_pse(self):
        f = self.file_h5py
        gPSE = f['PSE'] if 'PSE' in f else f.create_group('PSE')
        # params for spe
        self._replace_var_in_h5(gPSE,'hx',self.params['hx'])
        self._replace_var_in_h5(gPSE,'steps',self.params['steps'])
        # marched values
        self._replace_var_in_h5(gPSE,'q',self.q_PSE)
        self._replace_var_in_h5(gPSE,'x',self.x_PSE)
        self._replace_var_in_h5(gPSE,'alpha',self.alpha_PSE)
    def _refresh_pse_params(self):
        # upate PSE items
        if False: # get it from LST entries
            self.omega_PSE['text']=self.omega.get()
            self.beta_PSE['text'] = self.beta.get()
            self.Re_PSE['text'] = self.Re.get()
            self.ny_PSE['text'] = self.ny.get()
            self.y0_PSE['text'] = self.y0.get()
            self.y1_PSE['text'] = self.y1.get()
            self.lalpha_PSE['text'] = self.lalpha['text']
        else: # get it from self.params
            self.omega_PSE['text']=str(self.params['omega'])
            self.beta_PSE['text'] = str(self.params['beta'])
            self.Re_PSE['text'] = str(self.params['Re'])
            self.ny_PSE['text'] = str(self.params['ny'])
            self.y0_PSE['text'] = str(self.params['grid'].y.min())
            self.y1_PSE['text'] = str(self.params['grid'].y.max())
            #self.lalpha_PSE['text'] = '{:g}+{:g}j'.format(self.alpha0.real,self.alpha0.imag)
            self.lalpha_PSE['text'] = self.lalpha['text']
            self.set_text(self.hx,'{:g}'.format(1./self.alpha0.real))
            #self.hx['text'] = '{:g}'.format(1./self.alpha0.real)
    def _plot_spectrum(self):
        self.eigenvalues_line.set_xdata(self.eigenvalues.real)
        self.eigenvalues_line.set_ydata(self.eigenvalues.imag)
        #self.eigenvalues_line.set_linestyle('')
        self.eigenvalues_line.set_label(r'$\alpha$')
        self.ax_eigenvalues.relim() # recompute the ax_eigenvalues.dataLim
        self.ax_eigenvalues.autoscale_view() # update ax_eigenvalues.viewLim using the new dataLim
        #self.ax_eigenvalues.legend(loc='best',numpoints=1)
        self.ax_eigenvalues.set_xlabel(r'$\alpha_r$')
        self.ax_eigenvalues.set_ylabel(r'$\alpha_i$')
        self.ax_eigenvalues.set_xlim([-1,1])
        self.ax_eigenvalues.set_ylim([-1,1])
        # plot ubar
        self.ubar_line.set_xdata(self.baseflow.get_U())
        self.ubar_line.set_ydata(self.params['grid'].y)
        self.ax_ubar.relim() # recompute the ax_eigenvalues.dataLim
        self.ax_ubar.autoscale_view() # update ax_eigenvalues.viewLim using the new dataLim
        #self.ubar_line, = self.ax_ubar.plot(self.baseflow.get_U(),self.params['grid'].y)
        self.fig_eigenvalues.canvas.draw()
        #self.fig.tight_layout()


    def _replace_var_in_h5(self,group,varname,data):
        # delete variable if it exists in h5 file group
        if varname in group:
            del group[varname]
        # create dataset in group
        group.create_dataset(varname,data=data)
    def _read_var_in_h5(self,group,varname):
        if varname in group:
            return group[varname][()]
        else:
            return None
    def _save_spectrum(self):
        alpha = self.eigenvalues_line.get_xdata() + self.eigenvalues_line.get_ydata()*1.j
        if self.file==None:
            self.file = tk.filedialog.asksaveasfilename(filetypes=[('GUI h5 file','*.h5'),('all','*')])#parent=self.win)
            self.file_h5py = h5py.File(self.file,'a')
        f = self.file_h5py
        gLST = f['LST'] if 'LST' in f else f.create_group('LST')
        # save alphas on plot
        self._replace_var_in_h5(gLST,'alpha',self.eigenvalues)
        self._replace_var_in_h5(gLST,'eigenfuncr',self.eigenfuncr)
        #self._replace_var_in_h5(gLST,'eigenfuncl',self.eigenfuncl)
        # save parameters and wallnormal grid
        self._replace_var_in_h5(gLST,'omega',self.params['omega'])
        self._replace_var_in_h5(gLST,'Re',self.params['Re'])
        self._replace_var_in_h5(gLST,'beta',self.params['beta'])
        self._replace_var_in_h5(gLST,'ny',self.params['ny'])
        self._replace_var_in_h5(gLST,'y',self.params['grid'].y)
        # save baseflow
        gbaseflow = f['LST/baseflow'] if 'LST/baseflow' in f else f.create_group('LST/baseflow')
        self._replace_var_in_h5(gbaseflow,'U',self.baseflow.get_U())
        self._replace_var_in_h5(gbaseflow,'Uy',self.baseflow.get_Uy())
        self._replace_var_in_h5(gbaseflow,'Uyy',self.baseflow.get_Uyy())
        self._replace_var_in_h5(gbaseflow,'Ux',self.baseflow.get_Ux())
        self._replace_var_in_h5(gbaseflow,'V',self.baseflow.get_V())
        self._replace_var_in_h5(gbaseflow,'Vx',self.baseflow.get_Vx())
        self._replace_var_in_h5(gbaseflow,'Vy',self.baseflow.get_Vy())
        self._replace_var_in_h5(gbaseflow,'P',self.baseflow.get_P())
        # save diffs
        gdiffs = f['LST/diffs'] if 'LST/diffs' in f else f.create_group('LST/diffs')
        self._replace_var_in_h5(gdiffs,'Dy',self.diffs['Dy'])
        self._replace_var_in_h5(gdiffs,'Dyy',self.diffs['Dyy'])
        # save helper_mats
        ghelper_mats = f['LST/helper_mats'] if 'LST/helper_mats' in f else f.create_group('LST/helper_mats')
        self._replace_var_in_h5(ghelper_mats,'zero',self.helper_mats['zero'])
        self._replace_var_in_h5(ghelper_mats,'I',self.helper_mats['I'])
        self._replace_var_in_h5(ghelper_mats,'uvwP_from_LST',self.helper_mats['uvwP_from_LST'])
        self._replace_var_in_h5(ghelper_mats,'u_from_SPE',self.helper_mats['u_from_SPE'])
        self._replace_var_in_h5(ghelper_mats,'v_from_SPE',self.helper_mats['v_from_SPE'])
        self._replace_var_in_h5(ghelper_mats,'w_from_SPE',self.helper_mats['w_from_SPE'])
        self._replace_var_in_h5(ghelper_mats,'P_from_SPE',self.helper_mats['P_from_SPE'])
        self._replace_var_in_h5(ghelper_mats,'dLdomega',self.helper_mats['dLdomega'])
    def _save_id(self):
        ind = self.ind[0]
        alpha = self.eigenvalues_line.get_xdata()[ind] + self.eigenvalues_line.get_ydata()[ind]*1.j
        y = self.params['grid'].y
        ny = self.params['ny']
        qhat = self.eigenfuncr[:4*ny,ind]
        uhat = qhat[:ny]
        vhat = qhat[ny:2*ny]
        what = qhat[2*ny:3*ny]
        phat = qhat[3*ny:]
        if self.file==None:
            self.file = tk.filedialog.asksaveasfilename(filetypes=[('GUI h5 file','*.h5'),('all','*')])#parent=self.win)
            self.file_h5py = h5py.File(self.file,'a')
        f = self.file_h5py
        gLST = f['LST'] if 'LST' in f else f.create_group('LST')
        self._replace_var_in_h5(gLST,'eigenfuncr_index',ind)
        self._replace_var_in_h5(gLST,'eigenfuncr_alpha',alpha)
        self._replace_var_in_h5(gLST,'eigenfuncr_qhat',qhat)
        self._replace_var_in_h5(gLST,'eigenfuncr_uhat',uhat)
        self._replace_var_in_h5(gLST,'eigenfuncr_vhat',vhat)
        self._replace_var_in_h5(gLST,'eigenfuncr_what',what)
        self._replace_var_in_h5(gLST,'eigenfuncr_phat',phat)

    
    def _plot_id(self):
        # get data
        ind = self.ind[0]
        y = self.params['grid'].y
        ny = self.params['ny']
        self.qhat0 = self.eigenfuncr[:4*ny,ind]
        self.alpha0= self.eigenvalues[ind]
        uhat = self.qhat0[    :1*ny]
        vhat = self.qhat0[1*ny:2*ny]
        what = self.qhat0[2*ny:3*ny]
        phat = self.qhat0[3*ny:    ]
        # plot data
        self.uhatline1.set_xdata(uhat.real)
        self.uhatline1.set_ydata(y)
        self.uhatline2.set_xdata(uhat.imag)
        self.uhatline2.set_ydata(y)
        self.uhatline3.set_xdata(np.abs(uhat))
        self.uhatline3.set_ydata(y)
        self.vhatline1.set_xdata(vhat.real)
        self.vhatline1.set_ydata(y)
        self.vhatline2.set_xdata(vhat.imag)
        self.vhatline2.set_ydata(y)
        self.vhatline3.set_xdata(np.abs(vhat))
        self.vhatline3.set_ydata(y)
        self.whatline1.set_xdata(what.real)
        self.whatline1.set_ydata(y)
        self.whatline2.set_xdata(what.imag)
        self.whatline2.set_ydata(y)
        self.whatline3.set_xdata(np.abs(what))
        self.whatline3.set_ydata(y)
        self.phatline1.set_xdata(phat.real)
        self.phatline1.set_ydata(y)
        self.phatline2.set_xdata(phat.imag)
        self.phatline2.set_ydata(y)
        self.phatline3.set_xdata(np.abs(phat))
        self.phatline3.set_ydata(y)

        #self.eigenvalues_line.set_ydata(2 * np.sin(2 * np.pi * self.t))
        #self.eigenvalues_line.set_label(r'$2\sin{(2 \pi t)}$')
        #self.ax_eigenvalues.legend(loc='best',numpoints=1)
        for ax in [self.ax_uhat,self.ax_vhat,self.ax_what,self.ax_phat]:
            ax.relim() # recompute the ax.dataLim
            ax.autoscale_view() # update ax.viewLim using the new dataLim
        #self.fig.tight_layout()
        self.fig_eigenfunction.canvas.draw()

    def _solve(self):
        # get values from fields
        ny = int(self.ny.get())
        omega = float(self.omega.get())
        #F = omega*nu*10**6
        Re = float(self.Re.get())
        beta = float(self.beta.get())
        y0 = float(self.y0.get())
        y1 = float(self.y1.get())

        # create grid and solve LST
        nu=1./Re
        x_start = Re
        #y = np.linspace(y0,y1,ny)
        y = set_Cheby_mapped_y(y0,y1,ny)

        #set params for LST
        self.params={
                'Re':Re,
                'omega':omega,
                'beta':beta,
                'grid':grid1D_omega_beta_class(y=y),
                'ny':y.shape[0],
                'x_start':x_start,
                'x':x_start,
                'nu':nu,
                'alpha_closure_tol':1.E-12,
                'flags':{
                    #'LST2':True,
                    'LSTNP':True,
                    }
            }
        self.diffs={
                'Dy':set_D(y,d=1,order=4,output_full=True,uniform=False),
                'Dyy':set_D(y,d=2,order=4,output_full=True,uniform=False),
            }
        O = np.zeros((ny,ny))
        I = np.eye(ny)
        self.helper_mats = {
                'zero':O,
                'I':I,
                'uvwP_from_LST':np.block([[np.eye(4*ny),np.zeros((4*ny,4*ny))]]),
                'u_from_SPE':np.block([[I,O,O,O]]),
                'v_from_SPE':np.block([[O,I,O,O]]),
                'w_from_SPE':np.block([[O,O,I,O]]),
                'P_from_SPE':np.block([[O,O,O,I]]),
                }
        self.baseflow = blasius(self.params['grid'].y,x=self.params['x'],nu=self.params['nu'],plot=False)
        #self.baseflow = channel(y,x=self.params['x'],nu=nu,plot=False)

        self.helper_mats['dLdomega'] = dLdomega(self.params,self.diffs,self.helper_mats)

        if self.full_partial_var.get()=='full':
            print('solve value is ',self.full_partial_var.get())
            self.L,self.M,self.eigenvalues,self.eigenfuncl,self.eigenfuncr = LST(self.params,self.diffs,self.baseflow,self.helper_mats)
        elif self.full_partial_var.get()=='single':
            print('solve value is ',self.full_partial_var.get())
            alphas=[complex(self.eigenvalue_guess.get().replace(' ','')),]
            self.L,self.M,self.eigenvalues,self.eigenfuncl,self.eigenfuncr = LST_alphas(self.params,self.diffs,self.baseflow,self.helper_mats,alphas)
        else:
            print('SOMETHING IS WRONG!!! solve value is ',self.full_partial_var.get())



    def set_dropdown(self,master,dimoptions,default=None,row=0,column=0):
        var = tk.StringVar(master)
        #dimoptions = ['1D','2D','3D']
        if default==None:
            var.set(dimoptions[0])
            dim = tk.ttk.OptionMenu(self.fLST1,var,dimoptions[0],*dimoptions)
        else:
            var.set(default)
            dim = tk.ttk.OptionMenu(self.fLST1,var,default,*dimoptions)
        dim.grid(row=row,column=column)
        return var,dim
    #def text_graph(self,text,Label,**grid):
        #fig = plt.Figure(figsize=(0.2, 0.2),tight_layout=True,frameon=False)#, dpi=100)
        #ax = fig.add_subplot(111)
        #canvas = FigureCanvasTkAgg(fig,master=Label)
        #canvas.get_tk_widget().grid(**grid)
        #ax.text(0.2,0.2,text)
        ##ax.get_xaxis().set_visible(False)
        ##ax.get_yaxis().set_visible(False)
        #ax.axis('off')
        ##for item in [fig, ax]:
            ##item.patch.set_visible(False)
        #ax.patch.set_alpha(1.0)
        #fig.patch.set_alpha(1.0)
        #canvas.draw()
        #return canvas
    def update_annot(self,ind):
        #pos = matplotlib.collections.PathCollection(self.eigenvalues_line.get_path()).get_offsets()[ind["ind"][0]]
        pos = self.eigenvalues_line.get_xydata()[ind["ind"][0]]
        self.annot.xy = pos
        #text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                               #" ".join([names[n] for n in ind["ind"]]))
        text = "{}".format(" ".join(list(map(str,ind["ind"]))))
        self.annot.set_text(text)
        norm=plt.Normalize(1,4)
        cmap = plt.cm.RdYlGn
        #self.annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        self.annot.get_bbox_patch().set_alpha(0.4)

    def hover(self,event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax_eigenvalues:
            cont, ind = self.eigenvalues_line.contains(event)
            if cont:
                #print('shaun print ind',ind,True)
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig_eigenvalues.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig_eigenvalues.canvas.draw_idle()
    def onpick(self,event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind # get first point
        self.ind = ind
        points = np.array([xdata[ind], ydata[ind]]).flatten()
        print('onpick index:',ind)
        print('onpick points:', points)
        print('onpick points:      {:.10e}+{:.10e}'.format(points[0].real,points[1].real))
        print('onpick points:      {:.10e}+{:.10e}'.format(points[0].real*1.7208,points[1].real*1.7208))
        self.annotate.remove()
        self.annotate = self.ax_eigenvalues.annotate('index={: 4d}, data={:.2e},{:.2e}'.format(*ind,*points),xy=points,xytext=(0.5,1.01),textcoords='axes fraction',arrowprops=dict(facecolor='black',arrowstyle="-"))
        self.lid['text']='{: 4d}'.format(*ind)
        self.lalpha['text']='{:.2e}+{:.2e}i'.format(*points)
        #self.annotate.xy=points
        #self.,xytext=(0.5,1.01),textcoords='axes fraction',arrowprops=dict(facecolor='black',arrowstyle="-",connectionstyle="arc3"))
        #self.annotate.set_text('index={}, data={}'.format(ind,points))
        #self.annotate.arrow_patch.set_arrowstyle('-')
        
        self.fig_eigenvalues.canvas.draw()

    def on_key_press_eigenvalues(self,event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas_eigenvalues, self.toolbar_eigenvalues)
    def on_key_press_eigenfunction(self,event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas_eigenfunction, self.toolbar_eigenfunction)
    def on_key_press_PSE(self,event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas_PSE, self.toolbar_PSE)
    def _new(self):
        self.file = tk.filedialog.asksaveasfilename(filetypes=[('GUI h5 file','*.h5'),('all','*')])#parent=self.win)
        if self.file!='':  # if picked a file
            if self.file_h5py!=None:
                self.file_h5py.close() # close the old one
            self.file_h5py = h5py.File(self.file,'a') # open a new one
    def _saveas(self):
        self.file = tk.filedialog.asksaveasfilename(filetypes=[('GUI h5 file','*.h5'),('all','*')])#parent=self.win)
        if self.file!='':  # if picked a file
            if self.file_h5py!=None:
                self.file_h5py.close() # close the old one
            self.file_h5py = h5py.File(self.file,'a') # open a new one
            # save all data
            self._save()
    def _save(self):
        if self.file == None:
            self._saveas()
        else:
            # save all LST data
            self._save_spectrum()
            self._save_id()
            # save PSE data
            self._save_pse()
    def _open(self):
        #def set_text(entry,text):
            #entry.delete(0,tk.END)
            #entry.insert(0,text)
            #return

        self.file = tk.filedialog.askopenfilename(filetypes=[('GUI h5 file','*.h5'),('all','*')])#parent=self.win)
        if (self.file!='') and ('.h5' in self.file):
            self.file_h5py = h5py.File(self.file,'a')
            #with h5py.File(self.file,'r') as f:
            f = self.file_h5py

            # read gLST
            if 'LST' in f:
                gLST = f['LST']
                # read in spectrum and params
                self.eigenvalues = self._read_var_in_h5(gLST,'alpha')
                # plot eigenvalues on graph
                self.eigenfuncr = self._read_var_in_h5(gLST,'eigenfuncr')
                omega = self._read_var_in_h5(gLST,'omega')
                Re = self._read_var_in_h5(gLST,'Re')
                beta = self._read_var_in_h5(gLST,'beta')
                ny = self._read_var_in_h5(gLST,'ny')
                y = self._read_var_in_h5(gLST,'y')
                x_start = Re
                nu = 1./Re
                self.params={
                        'Re':Re,
                        'omega':omega,
                        'beta':beta,
                        'grid':grid1D_omega_beta_class(y=y),
                        'ny':y.shape[0],
                        'x_start':x_start,
                        'x':x_start,
                        'nu':nu,
                        'alpha_closure_tol':1.E-12,
                        'flags':{
                            'LST2':True,
                            }
                    }
                # set spectrum and params
                self.set_text(self.omega,str(omega))
                self.set_text(self.Re,str(Re))
                self.set_text(self.beta,str(beta))
                self.set_text(self.ny,str(ny))
                self.set_text(self.y0,str(y.min()))
                self.set_text(self.y1,str(y.max()))
                # read in baseflow
                gbaseflow = f['LST/baseflow']
                self.baseflow = base_flow_class(
                        U = self._read_var_in_h5(gbaseflow,'U'),
                        Uy = self._read_var_in_h5(gbaseflow,'Uy'),
                        Uyy = self._read_var_in_h5(gbaseflow,'Uyy'),
                        Ux = self._read_var_in_h5(gbaseflow,'Ux'),
                        V = self._read_var_in_h5(gbaseflow,'V'),
                        Vy = self._read_var_in_h5(gbaseflow,'Vy'),
                        Vx = self._read_var_in_h5(gbaseflow,'Vx'),
                        P = self._read_var_in_h5(gbaseflow,'P'))
                self._plot_spectrum()
                # read diffs
                gdiffs = f['LST/diffs']
                self.diffs = {
                        'Dy':self._read_var_in_h5(gdiffs,'Dy'),
                        'Dyy':self._read_var_in_h5(gdiffs,'Dyy') }
                # save helper_mats
                ghelper_mats = f['LST/helper_mats']
                self.helper_mats = {
                        'zero':self._read_var_in_h5(ghelper_mats,'zero'),
                        'I':self._read_var_in_h5(ghelper_mats,'I'),
                        'uvwP_from_LST':self._read_var_in_h5(ghelper_mats,'uvwP_from_LST'),
                        'u_from_SPE':self._read_var_in_h5(ghelper_mats,'u_from_SPE'),
                        'v_from_SPE':self._read_var_in_h5(ghelper_mats,'v_from_SPE'),
                        'w_from_SPE':self._read_var_in_h5(ghelper_mats,'w_from_SPE'),
                        'P_from_SPE':self._read_var_in_h5(ghelper_mats,'P_from_SPE'),
                        'dLdomega':self._read_var_in_h5(ghelper_mats,'dLdomega') }
                # read in eigenfunction id
                self.ind = [self._read_var_in_h5(gLST,'eigenfuncr_index'),]
                self._plot_id() # plot specified index
                self.alpha0 = self._read_var_in_h5(gLST,'eigenfuncr_alpha')
                self.qhat0 = self._read_var_in_h5(gLST,'eigenfuncr_qhat')
                id_uhat = self._read_var_in_h5(gLST,'eigenfuncr_uhat')
                id_vhat = self._read_var_in_h5(gLST,'eigenfuncr_vhat')
                id_what = self._read_var_in_h5(gLST,'eigenfuncr_what')
                id_phat = self._read_var_in_h5(gLST,'eigenfuncr_phat')
                
                # draw eigenvalues_line to index
                xdata = self.eigenvalues_line.get_xdata()
                ydata = self.eigenvalues_line.get_ydata()
                ind = self.ind # get first point
                points = np.array([xdata[ind], ydata[ind]]).flatten()
                print('onpick index:',ind)
                print('onpick points:', points)
                self.annotate.remove()
                self.annotate = self.ax_eigenvalues.annotate('index={: 4d}, data={:.2e},{:.2e}'.format(*ind,*points),xy=points,xytext=(0.5,1.01),textcoords='axes fraction',arrowprops=dict(facecolor='black',arrowstyle="-"))
                self.lid['text']='{: 4d}'.format(*ind)
                self.lalpha['text']='{:.2e}+{:.2e}i'.format(*points)
                #self.annotate.xy=points
                #self.,xytext=(0.5,1.01),textcoords='axes fraction',arrowprops=dict(facecolor='black',arrowstyle="-",connectionstyle="arc3"))
                #self.annotate.set_text('index={}, data={}'.format(ind,points))
                #self.annotate.arrow_patch.set_arrowstyle('-')
                self.fig_eigenvalues.canvas.draw()
            else:
                gLST = None # no LST data
                print('no LST data')
            print('x = ',self.params['x'])
            if 'PSE' in f:
                gPSE = f['PSE']
                # open params
                self.params['hx'] = self._read_var_in_h5(gPSE,'hx')
                self.params['steps'] = self._read_var_in_h5(gPSE,'steps')
                # open marched values
                self.q_PSE = self._read_var_in_h5(gPSE,'q')
                self.x_PSE = self._read_var_in_h5(gPSE,'x')
                self.alpha_PSE = self._read_var_in_h5(gPSE,'alpha')
                # update text to match LST
                self.omega_PSE['text'] = self.omega.get()
                self.beta_PSE['text'] = self.beta.get()
                self.Re_PSE['text'] = self.Re.get()
                self.ny_PSE['text'] = self.ny.get()
                self.y0_PSE['text'] = self.y0.get()
                self.y1_PSE['text'] = self.y1.get()
                self.lalpha_PSE['text'] = self.lalpha['text']
                self.set_text(self.hx,str(self.params['hx']))
                self.set_text(self.steps,str(self.params['steps']))
                # plot pse
                self._plot_pse()
    def _quit(self):
        if (self.file!=None) and (self.file!='') and ('.h5' in self.file):
            self.file_h5py.close()
        self.win.quit()     # stops mainloop
        self.win.destroy()  # this is necessary on Windows to prevent
                       # Fatal Python Error: PyEval_RestoreThread: NULL tstate
        
def GUI():
    #window=tk.Tk()
    window=ttkthemes.ThemedTk(theme='clearlooks',background=True)
    mywin=MyWindow(window)
    #window.title('Hello Python')
    #window.geometry("400x300+10+10")
    #style=ttkthemes.ThemedStyle(window)
    #style.set_theme('clearlooks')

    window.mainloop()

if __name__=="__main__":
    GUI()
