import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os, shutil
import tkinter as tk

# Diferencijalna jednacina i njeni parametri
# m * u'' + Mi * u' + ku = 0
# mass - oscilator mass (Shouldnt be negative)
# Mi - friction coef
# k - spring const
# x0 - Starting position 
# v0 - Starting velocity (positive is up)

duration = 10       # Duzina trajanja simulacije oscilovanja. POTENCIJALNI ULAZNI PARAMETAR


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
    def __init__(self, mass, Mi, k, x0 ,v0):
        self.mass = mass        # Object mass
        self.Mi = Mi        # Friction coef
        self.k = k      # Spring coef
        self.x0 = x0        # Starting position
        self.v0 = v0        # Starting velocity (+ up / - down)
        self.interval = dde.geometry.TimeDomain(0, duration)     # Resava se na domenu t=(0,duration)
        self.delta = Mi / 2*mass
        self.w0 = np.sqrt(k / mass)

    # Funkcija za treniranje i predvidjanje
    # Prima neophodne parametre za podesavanje i treniranje mreze koje korisnik moze da podesava odvojeno od ulaznih parametara
    def TrainModel(self, numDomain, numBoundary, numTest, layers, numEpoch):
        # Egzaktno resenje
        def func(t):
            #print("sssssssssssss ", delta, w0)
            
            #delta = Mi / 2*mass
            #w0 = np.sqrt(k / mass)
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
                return np.exp(-self.delta*t) * (self.x0 + (self.v0 + self.delta*x0) * t)

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
        
        for i in range(5):
            # Trening podaci
            data = dde.data.TimePDE(self.interval, Ode, [ic1, ic2], numDomain, numBoundary, solution=func, num_test=numTest)

            # kompajliranje modela
            model = dde.Model(data, net)
            model.compile("adam", lr=.001, metrics=["l2 relative error"])
            
            losshistory, train_state = model.train(epochs=numEpoch, callbacks=[])
            
            rmse = np.sqrt(np.sum((train_state.y_test - train_state.best_y)**2))
            
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
        pass
    
    def TkMain(self):
        mainWindow = tk.Tk()
        mainWindow.geometry("800x600")
        
        frameSettings = tk.Frame(master=mainWindow).grid(row=0,column=0)
        frameSimulation = tk.Frame()
        frameGraph = tk.Frame()
        
        l_mass = tk.Label(master=frameSettings, text="Mass", width=150, height=35)
        l_mi = tk.Label(master=frameSettings, text="Friction", width=150, height=35)
        l_k = tk.Label(master=frameSettings, text="Spring", width=150, height=35)
        
        in_mass = tk.Entry(master=frameSettings, width=150)
        in_mi = tk.Entry(master=frameSettings, width=150)
        in_k = tk.Entry(master=frameSettings, width=150)
        
        btn_begin = tk.Button(master=frameSettings, text="Begin simulation")
        
        #l_mass.place(x=20,y=50)
        #in_mass.place(x=20, y=90)
        l_mass.grid(row=0, column=0)
        in_mass.grid(row=1, column=0)
        l_mi.grid(row=3, column=0)
        in_mi.grid(row=4, column=0)
        l_k.grid(row=6, column=0)
        in_k.grid(row=7, column=0)
        
        mainWindow.mainloop()

# Pomocna funkcija za spajanje nisa x i y podataka
# Koristi se kod iscrtavanja grafika
def PackData(x, y):
    data = []

    for i in range(len(x)):
        data += [(x[i][0], y[i][0])]
        
    return data


# ======= MAIN ==============
if __name__ == "__main__":
    print("Mass")
    mass = float(input())
    print("Friction")
    Mi = float(input())
    print("Spring")
    k = float(input())
    print("Starting position")
    x0 = float(input())  # Starting position
    print("Starting velocity")
    v0 = float(input())  # Starting velocity
    
    pinn = PINN(mass, Mi, k, x0, v0)
    losshistory, train_state = pinn.TrainModel(100, 20, 200,  [1] + [30] * 2 + [1], 5000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    # train = PackData(train_state.X_train, train_state.y_train)
    # train = sorted(train, key=lambda x: x[0])

    # test = PackData(train_state.X_test, train_state.y_test)
    # test = sorted(test, key=lambda x: x[0])

    # pred = PackData(train_state.X_test, train_state.best_y)
    # pred = sorted(pred, key=lambda x: x[0])

    # shutil.rmtree("images")
    # os.mkdir("images")

    # plt.xlim((-1,duration+1))
    
    # plt.plot([x[0] for x in test], [x[1] for x in test], color="blue")
    # plt.show()

    # # for i in range(len(test)):
    # #     plt.plot(pred[i][0], pred[i][1], color="red", linestyle="dashed", linewidth=2, marker=".")
    # #     plt.legend(["actual", "predicted"])
    # #     plt.savefig(f"images/{i}.png")
    # #     print(f"Progress: {int((i / len(test)) * 100)}%")
    
    #gui = GUI()
    #gui.TkMain()