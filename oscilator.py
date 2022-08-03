from turtle import width
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import threading, os
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# Diferencijalna jednacina i njeni parametri
# m * u'' + Mi * u' + ku = 0
# mass - oscilator mass (Shouldnt be negative)
# Mi - friction coef
# k - spring const
# x0 - Starting position 
# v0 - Starting velocity (positive is up)

#duration = 10       # Duzina trajanja simulacije oscilovanja. POTENCIJALNI ULAZNI PARAMETAR


# Testni slucajevi
# delta > w0

# mass = 0.2
# Mi = 10
# k = 0.01

# detla == w0

#mass = 1
#Mi = 2
#k = 1

# delta < w0

# mass = 1
# Mi = 0.5
# k = 1

#x0 = 0  # Starting position
#v0 = 2  # Starting velocity

#delta = Mi / 2*mass
#w0 = np.sqrt(k / mass)

class PINN: 
    # Prima ulazne parametre
    def __init__(self, mass, Mi, k, x0 ,v0, duration):
        self.mass = mass        # Object mass
        self.Mi = Mi        # Friction coef
        self.k = k      # Spring coef
        self.x0 = x0        # Starting position
        self.v0 = v0        # Starting velocity (+ up / - down)
        self.interval = dde.geometry.TimeDomain(0, duration)     # Resava se na domenu t=(0,duration)
        self.delta = Mi / 2*mass
        self.w0 = np.sqrt(k / mass)
        
    # Pomocna funkcija za spajanje nisa x i y podataka
    # Koristi se kod iscrtavanja grafika
    def PackData(self, x, y):
        data = []

        for i in range(len(x)):
            data += [(x[i][0], y[i][0])]
            
        return data

    # Funkcija za treniranje i predvidjanje
    # Prima neophodne parametre za podesavanje i treniranje mreze koje korisnik moze da podesava odvojeno od ulaznih parametara
    def TrainModel(self, numDomain, numBoundary, numTest, layers, numEpoch):
        # Egzaktno resenje
        def func(t):
            #print("sssssssssssss ", delta, w0)
            
            # delta = Mi / 2*mass - NIJE DEO KODA
            # w0 = np.sqrt(k / mass) - NIJE DEO KODA
            # Postoje 3 slucaja za dobijanje egzaktnog resenja koji potencijalno sadrze svoje podslucajeve:
            
            
            # OVERDAMBED CASE   - imamo jaku silu trenja
            # Ovaj slucaj zvacicno ima 3 podslucajeva koja ne koristimo jer nisu moguca ukoliko je delta > w0
            # tako da koristimo samo jedno resenje
            if(self.delta > self.w0):
                alpha = np.sqrt(self.delta*self.delta - self.w0*self.w0)

                c1 = (self.v0 + self.x0 * (self.delta + alpha)) / (2 * alpha)
                c2 = -(self.v0 + self.x0 * (self.delta - alpha)) / (2 * alpha)
                
                
                return np.exp(-self.delta*t) * (c1 * np.exp(alpha * t) + c2 * np.exp(-alpha * t))

            # CRITICALLY DAMPED CASE
            # Ovaj slucaj je najjednostavniji od svih
            elif(self.delta == self.w0):
                return np.exp(-self.delta*t) * (self.x0 + (self.v0 + self.delta*self.x0) * t)

            # UNDERDAMBED CASE
            # Najcesci slucaj gde se najbolje i prikazuje oscilovanje
            # Ovaj slucaj sadrzi vise podslucajeva ali dva su dovoljna
            else:
                w = np.sqrt(self.w0*self.w0 - self.delta*self.delta)
                n = 2
                if(self.delta == 0 and self.v0 == 0): # Harmonijske
                    A = -self.x0 / 2
                    fi = (2 * np.pi * n) + np.pi
                elif(self.delta != 0):       # Neprigusene oscilacije
                    A = -self.v0 / (2 * self.delta)
                    fi = (2 * np.pi * n) + np.pi
                
                return np.exp(-self.delta * t) * (2*A*np.cos(fi + w*t))
                

        # Da li je tacka blizu t=0
        def boundary_l(t, on_initial):
            return on_initial and np.isclose(t[0], 0)

        # Jednacina ODE
        def Ode(x,y):
                dyX = dde.grad.jacobian(y,x)        # Prvi izvod
                dyXX = dde.grad.hessian(y,x)        # Drugi izvod
                return self.mass * dyXX + self.Mi * dyX + self.k * y
            
        # Inicijalni slucajevi
        # y(0) = x0
        def bc_func1(inputs, outputs, X):
            return outputs - self.x0

        # y'(0) = v0
        def bc_func2(inputs, outputs, X):
            return dde.grad.jacobian(outputs, inputs, i=0,j=None) - self.v0

        ic1 = dde.icbc.OperatorBC(self.interval, bc_func1, boundary_l)
        ic2 = dde.icbc.OperatorBC(self.interval, bc_func2, boundary_l)


        
        # Podesavanje mreze  
        #layers = [1] + [30] * 2 + [1]
        activation = "tanh"
        init = "Glorot uniform"
        net = dde.nn.FNN(layers, activation, init)
        
        # Treniranje modela
        # Zbog toga sto su trening podaci nasumicni, nalazenje najboljeg resenja je otezano
        # Zato generisanje podataka i treniranje vrsimo vise puta i kao rezultat uzimamo najbolje resenje
        # Rezultate gledamo na osnovu RMSE metrike
        
        bestRMSE = float("inf")
        
        for i in range(1):
            # Trening podaci
            data = dde.data.TimePDE(self.interval, Ode, [ic1, ic2], numDomain, numBoundary, solution=func, num_test=numTest)

            # kompajliranje modela
            model = dde.Model(data, net)
            model.compile("adam", lr=.001, metrics=["l2 relative error"])
            
            losshistory, train_state = model.train(epochs=numEpoch, callbacks=[])
            
            rmse = np.sqrt(np.sum((train_state.y_test - train_state.best_y)**2))
            
            if(i == 0):
                bestRMSE = rmse
                bestTrain = train_state
                bestHist = losshistory
                
            if(rmse < bestRMSE):
                bestRMSE = rmse
                bestTrain = train_state
                bestHist = losshistory
            
            #print(f"{i+1} RMSE: {np.sqrt(np.sum((train_state.y_test - train_state.best_y)**2))}")
        
        #print(f"BEST RMSE: {bestRMSE}") 
        return bestHist, bestTrain

# GUI klasa

class GUI:
    def __init__(self):
        ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
        ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

        self.app = ctk.CTk()  # create CTk window like you do with the Tk window
        self.app.geometry("1100x720")
        
        self.mass = tk.StringVar()
        self.mi = tk.StringVar()
        self.k = tk.StringVar()
        self.position = tk.StringVar()
        self.velocity = tk.StringVar()
        self.domain = tk.StringVar()
        self.bounds = tk.StringVar()
        self.tests = tk.StringVar()
        self.epochs = tk.StringVar()
        self.time = tk.StringVar()
        
        self.latestPred = [(0,0)]
        
        self.frameSettings = ctk.CTkFrame(self.app, height=700)
        self.frameSimulation = ctk.CTkFrame(self.app, width=400, height=700)
        self.frameGraph = ctk.CTkFrame(self.app, width=440, height=700)
        self.frameTrainingSettings = ctk.CTkFrame(self.app, height=700)
        
    def DrawGraph(self, data):
        figure1 = plt.Figure(figsize=(9,5), dpi=90)
        ax1 = figure1.add_subplot(111)
        plot = FigureCanvasTkAgg(figure1, self.frameGraph)
        plot.get_tk_widget().grid(row=1, column=0, rowspan=14)
        ax1.plot([x[0] for x in data], [x[1] for x in data])
        ax1.set_title('Prediction')
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Distance [cm]")
        
    def DrawSimulation(self, playButton, data=[(0,0)]):
        playButton.configure(state=tk.DISABLED)
        canvas = ctk.CTkCanvas(self.frameSimulation, width=400, height=580,bg="white")
        canvas.place(x=0, y=80)
        
        # -- STATIC OBJECTS --
        canvas.create_line(0,200,400,200, dash=(10,1))      # zero line
        
        for x in range(0,400,10):       # anchor suface 1
            canvas.create_line(x,50,x+8,35)
        canvas.create_line(0,50,400,50,width=3)         # anchor surface 2
        
        temp = -2
        canvas.create_line(10,50,10,600, width=2)       # height line 1
        for x in range(100,600,50):      # height line 2
            canvas.create_line(10,x,20,x)
            canvas.create_text(25, x, text=temp)
            temp += 1
        
        
        # -- MOVING OBJECTS -- 
        massHeightPos = 200 - 50 * data[0][1]
        massBlock = canvas.create_polygon(150,massHeightPos,250,massHeightPos, 250,massHeightPos+100,150,massHeightPos+100)     # mass block 
        spring = canvas.create_line(200,50,200,massHeightPos)        # spring
        
        for x in data:
            canvas.delete(massBlock)
            canvas.delete(spring)
            
            massHeightPos = 200 - 50 * x[1]
            massBlock = canvas.create_polygon(150,massHeightPos,250,massHeightPos, 250,massHeightPos+100,150,massHeightPos+100)     # mass block
            
            spring = canvas.create_line(200,50,200,massHeightPos)        # spring
            
            canvas.after(5)
        
        playButton.configure(state=tk.NORMAL)
        
    def StartTrainingThread(self, startButton, playButton):
        
        pinn = PINN(float(self.mass.get()), float(self.mi.get()), float(self.k.get()), float(self.position.get()), float(self.velocity.get()), float(self.time.get()))
        losshistory, train_state = pinn.TrainModel(int(self.domain.get()), int(self.bounds.get()), int(self.tests.get()),  [1] + [30] * 2 + [1], int(self.epochs.get()))
        #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        
        pred = pinn.PackData(train_state.X_test, train_state.best_y)
        pred = sorted(pred, key=lambda x: x[0])

        self.DrawGraph(pred)
        
        self.latestPred = pred
        self.DrawSimulation(playButton, pred)
        
        startButton.configure(state=tk.NORMAL)
        
    def ReplaySimulationThread(self, playButton, data):
        thread = threading.Thread(target=self.DrawSimulation, args=(playButton, data,))
        thread.start()
        
    def StartSimulation(self, startButton, playButton):
        startButton.configure(state=tk.DISABLED)
        thread = threading.Thread(target=self.StartTrainingThread, args=(startButton, playButton,))
        thread.start()
        
        #button.configure(state=tk.NORMAL)
    
    def TkMain(self):
        
        def CheckIfNumber(char):
            if(char != ""):
                try:
                    float(char)
                except:
                    return False

            return True
        
        def CheckIfPositiveNumber(char):
            if(char != ""):
                try:
                    if(float(char) <= 0):
                        return False
                except:
                    return False

            return True
        
        def HideSettings():
            self.frameTrainingSettings.tkraise()
            
        def ShowSettings():
            self.frameSettings.tkraise()
        
        # -- FRAME SETTINGS --
        
        self.frameSettings.columnconfigure((0), weight=1)
        self.frameSettings.rowconfigure(([x for x in range(15)]), weight=1)
        self.frameSettings.grid(row=0,column=0, sticky=tk.NS, padx=10, pady=10)
        self.frameSettings.grid_propagate(0)
        
        self.frameSimulation.columnconfigure((0), weight=1)
        self.frameSimulation.rowconfigure(([x for x in range(15)]), weight=1)
        self.frameSimulation.grid(row=0,column=1, sticky=tk.NS, padx=10, pady=10)
        self.frameSimulation.grid_propagate(0)
        
        self.frameGraph.columnconfigure((0), weight=1)
        self.frameGraph.rowconfigure(([x for x in range(15)]), weight=1)
        self.frameGraph.grid(row=0,column=2, sticky=tk.NS, padx=10, pady=10)
        self.frameGraph.grid_propagate(0)
        
        self.frameTrainingSettings.columnconfigure((0), weight=1)
        self.frameTrainingSettings.rowconfigure(([x for x in range(15)]), weight=1)
        self.frameTrainingSettings.grid(row=0,column=0, sticky=tk.NS, padx=10, pady=10)
        self.frameTrainingSettings.grid_propagate(0)
        
        self.frameSettings.tkraise()
        
        # -- WIDGETS -- 
        
        imagePath = os.path.join("Resources", "SettingsButtonImg.png")
        image = Image.open(imagePath).resize((15,15), Image.ANTIALIAS)
        SettingsPhoto = ImageTk.PhotoImage(image)
        
        imagePath = os.path.join("Resources", "resetButtonImg.png")
        image = Image.open(imagePath).resize((15,15), Image.ANTIALIAS)
        ResetPhoto = ImageTk.PhotoImage(image)
        
        l_mass = ctk.CTkLabel(master=self.frameSettings, text="Mass", width=150, height=35)
        l_mi = ctk.CTkLabel(master=self.frameSettings, text="Friction", width=150, height=35)
        l_k = ctk.CTkLabel(master=self.frameSettings, text="Spring", width=150, height=35)
        l_position = ctk.CTkLabel(master=self.frameSettings, text="Starting position", width=150, height=35)
        l_velocity = ctk.CTkLabel(master=self.frameSettings, text="Starting velocity", width=150, height=35)
        l_titleSettings = ctk.CTkLabel(master=self.frameSettings, text="PHYSICAL PARAMETERS", width=150, height=35)
        l_titleSimulation = ctk.CTkLabel(master=self.frameSimulation, text="SIMULATION", width=150, height=35)
        l_titleGraph = ctk.CTkLabel(self.frameGraph, text="GRAPH", width=150, height=35)
        l_titleModelSettings = ctk.CTkLabel(master=self.frameTrainingSettings, text="PINN PARAMETERS", width=150, height=35)
        
        l_numDomain = ctk.CTkLabel(self.frameTrainingSettings, text="Domain points", width=150, height=35)
        l_numBounds = ctk.CTkLabel(self.frameTrainingSettings, text="Boundary points", width=150, height=35)
        l_numTest = ctk.CTkLabel(self.frameTrainingSettings, text="Number of tests", width=150, height=35)
        l_numEpochs = ctk.CTkLabel(self.frameTrainingSettings, text="Number of epochs", width=150, height=35)
        l_time = ctk.CTkLabel(self.frameTrainingSettings, text="Simulation duration [s]", width=150, height=35)
        
        numValidator = self.app.register(CheckIfNumber)
        positiveNumValidator = self.app.register(CheckIfPositiveNumber)
        
        in_mass = ctk.CTkEntry(master=self.frameSettings, width=150, textvariable=self.mass, validate="key", validatecommand=(positiveNumValidator, "%P"), justify="right")
        in_mi = ctk.CTkEntry(master=self.frameSettings, width=150, textvariable=self.mi, validate="key", validatecommand=(numValidator, "%P"), justify="right")
        in_k = ctk.CTkEntry(master=self.frameSettings, width=150, textvariable=self.k, validate="key", validatecommand=(numValidator, "%P"), justify="right")
        in_position = ctk.CTkEntry(master=self.frameSettings, textvariable=self.position, width=150, validate="key", validatecommand=(numValidator, "%P"), justify="right")
        in_velocity = ctk.CTkEntry(master=self.frameSettings, textvariable=self.velocity, width=150, validate="key", validatecommand=(numValidator, "%P"), justify="right")
        
        in_numDomain = ctk.CTkEntry(master=self.frameTrainingSettings, width=150, textvariable=self.domain, validate="key", validatecommand=(positiveNumValidator, "%P"), justify="right")
        in_numBounds = ctk.CTkEntry(master=self.frameTrainingSettings, width=150, textvariable=self.bounds, validate="key", validatecommand=(positiveNumValidator, "%P"), justify="right")
        in_numTest = ctk.CTkEntry(master=self.frameTrainingSettings, width=150, textvariable=self.tests, validate="key", validatecommand=(positiveNumValidator, "%P"), justify="right")
        in_numEpochs = ctk.CTkEntry(master=self.frameTrainingSettings, width=150, textvariable=self.epochs, validate="key", validatecommand=(positiveNumValidator, "%P"), justify="right")
        in_time = ctk.CTkEntry(master=self.frameTrainingSettings, width=150, textvariable=self.time, validate="key", validatecommand=(positiveNumValidator, "%P"), justify="right")
        
        in_numDomain.insert(tk.END, "100")
        in_numBounds.insert(tk.END, "20")
        in_numTest.insert(tk.END, "200")
        in_numEpochs.insert(tk.END, "5000")
        in_time.insert(tk.END, "10")
        
        btn_playSimulation = ctk.CTkButton(master=self.frameSimulation, text="Play")
        btn_playSimulation.configure(command=lambda x=btn_playSimulation:self.ReplaySimulationThread(x, self.latestPred))
        btn_begin = ctk.CTkButton(master=self.frameSettings, text="Start")
        btn_begin.configure(command=lambda x=btn_begin, y=btn_playSimulation:self.StartSimulation(x,y))
        btn_swapSettings = ctk.CTkButton(master=self.frameTrainingSettings,text="", image=SettingsPhoto, command=ShowSettings, width=15, height=15).place(x=10,y=10)
        btn_swapTrainSettings = ctk.CTkButton(master=self.frameSettings,text="", image=SettingsPhoto, command=HideSettings, width=15, height=15).place(x=10,y=10)
        
        
        self.DrawSimulation(btn_playSimulation)
        #btn_playSimulation.configure(state=tk.DISABLED)
        
        # -- PLACEMENT -- 
        
            # -- FrameSettings
        
        l_titleSettings.grid(row=0, column=0)
        
        l_mass.grid(row=2, column=0)
        in_mass.grid(row=3, column=0)
        
        l_mi.grid(row=4, column=0)
        in_mi.grid(row=5, column=0)
        
        l_k.grid(row=6, column=0)
        in_k.grid(row=7, column=0)
        
        l_position.grid(row=8, column=0)
        in_position.grid(row=9, column=0)

        l_velocity.grid(row=10, column=0)
        in_velocity.grid(row=11, column=0)
        
        btn_begin.grid(row=13, column=0)
 
            # -- FrameTrainSettings -- 
            
        l_titleModelSettings.grid(row=0, column=0)
 
        l_numDomain.grid(row=2, column=0)
        in_numDomain.grid(row=3, column=0)
        
        l_numBounds.grid(row=4, column=0)
        in_numBounds.grid(row=5, column=0)
        
        l_numTest.grid(row=6, column=0)
        in_numTest.grid(row=7, column=0)
        
        l_numEpochs.grid(row=8, column=0)
        in_numEpochs.grid(row=9, column=0)
        
        l_time.grid(row=10, column=0)
        in_time.grid(row=11, column=0)
        
            # -- FrameSimulation --
        
        l_titleSimulation.grid(row=0, column=0)
        
        btn_playSimulation.grid(row=15, column=0)
        
            # -- FrameGraph --
        
        l_titleGraph.grid(row=0, column=0)
        
        self.app.mainloop()


# ======= MAIN ==============
if __name__ == "__main__":
    gui = GUI()
    gui.TkMain()