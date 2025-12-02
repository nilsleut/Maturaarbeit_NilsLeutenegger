"""
Dieses Skript wurde im Rahmen einer Maturaarbeit entwickelt.
Es implementiert ein eigenständiges Multi-Layer-Perceptron (MLP)
mit verschiedenen Optimierungsverfahren.
"""


import numpy as np
import scipy.special
import copy


class MLP:
    """
    Unterstützte Merkmale:
    - Verschiedene Aktivierungsfunktionen: ReLU, Sigmoid
    - Verlustfunktionen: MSE, Huber-Loss
    - Regularisierung: L2, L1 oder keine
    - Dropout auf versteckten Schichten
    - Mehrere Optimierer: SGD (+Momentum / Nesterov), AdaGrad, RMSProp, Adam
      sowie eine Conjugate-Gradient-Variante mit Line-Search
    - Early-Stopping über Validierungsset (Patience-Parameter)
    - Gewichtsinitalisierungen (He / Xavier) abgestimmt auf Aktivierungsfunktion

    Wichtige Methoden:
    - fit(X, Y): Trainiert das Modell an Trainingsdaten und validiert intern
    - predict(X): Liefert Vorhersagen für Eingabedaten X
    """

    def __init__(
        self, 
        akt="relu", 
        hiddenlayer=(50, 50, 50, 50), 
        lossfunction = "MSE",
        huberdelta = 1.0,  
        dropout=0.5, 
        eta=0.001, 
        maxIter=200, 
        regul="l2", 
        regulrate = 0.001, 
        patience = 250, 
        batchsize = 1, 
        sgdmaxiterations = 300, 
        momentum = 0.9, 
        sgdupdate = False, 
        nestrov = False, 
        adagrad = False, 
        adagradconst = 10e-7, 
        RMSProp = True, 
        RMSPropconst = 10e-6, 
        RMSProprate = 0.9, 
        Adam = True, 
        firstmomentrate = 0.9, 
        secmomentrate = 0.999, 
        Adamconst = 10e-8, 
        conjugateGradient = True, 
        alpha = 0.01, 
        c1 = 0.0001, 
        c2 = 0.9, 
        rho = 0.5, 
        maxlinesearch = 20,
        ):


        #Arichtekturparameter
        self.hl = hiddenlayer 
        self.xMin = 0.0
        self.xMax = 1.0
        self.akt = akt

        #Aktivierungsfunktion mit dazugehörigen Ableitungen
        if akt == "relu":
            self._aktivierung = lambda x: np.maximum(0, x)
            self._ableit = lambda x: (x > 0).astype(float)
        elif akt == "sigmoid":
            self._aktivierung = lambda x: scipy.special.expit(x)
            self._ableit = lambda x: scipy.special.expit(x) * (1 - scipy.special.expit(x))
        
        #Verlustfunktion
        self.lossfunction = lossfunction
        self.huberdelta = huberdelta
        
        #Regularisierungsparameter
        self.dropout = dropout #Dropoutrate (0 = kein Dropout)
        self.eta = eta #Learnrate
        self.maxIter = maxIter #Max Iterationen
        self.regul = regul #Normparameter Strafe (L2, L1, No)
        self.regulrate = regulrate #Regularisierungsrate (0 = keine Regularisierung)
        self.patience = patience #Maximale Dauer ohne Verbesserung (hohe Nummer = kein early stopping)
        self.trainingsteps = 0 


        #Optimizerunsparameterparameter
        self.batchsize = batchsize #Grösse jedes Batches (0 = ganzen Batch verwenden)
        self.sgdmaxiterations = sgdmaxiterations #Max Dauer des SGD Updates
        self.momentum = momentum #Rate für momentum (0 = kein momentum)
        self.sgdupdate = sgdupdate #sgd update verwenden
        self.nestrov = nestrov #nestrov momentum verwenden
        self.adagradconst = adagradconst #kleine Konstante für numerische Stabilität 
        self.adagrad = adagrad #AdaGrad verwenden
        self.RMSProp = RMSProp #RMSProp verwenden
        self.RMSPropconst = RMSPropconst #kleine Konstante für numerische Stabilität
        self.RMSProprate = RMSProprate #RMSProp Verfallsrate 
        self.Adam = Adam #Adam verwenden
        self.firstmomentrate = firstmomentrate #first moment Verfallsrate
        self.secmomentrate  = secmomentrate #second moment Verfallsrate
        self.Adamconst = Adamconst #kleine Konstante für numerische Stabilität
        self.conjugateGradient = conjugateGradient #Newtons Method
        

        #Second order Methoden (cg) Parameter
        self.alpha = alpha
        self.c1 = c1 #Armijo constant
        self.c2 = c2 # Wolfe constant
        self.rho = rho #step size reduction factor
        self.maxlinesearchiters = maxlinesearch


        #Speicher für Trainingsverlauf
        self.W = [] #Gewichtsmatrizen
        self.epoch = [] #Verlauf der Iterationsschritte
        self.meanE = [] #Verlauf des Validierungsfehlers
        self.yp = 0 #letzte Vorhersage auf Validierungsset
        self.YVal = 0 #letzte echten Werte des Validierungssets
        self.meanEvar = 1000 #aktueller Validierungsfehler
        self.modelused = "" #verwendeter Optimierer
        self.dropoutused = False #Flag, ob Dropout verwendet wurde
        self.earlystop = False #Flag für early stopping
        self.minError = float('inf') #Minimaler Validierungsfehler

    def _initWeights(self):
        """
        Initialisiert die Gewichtsmatrizen jeder Schicht des Netzwerks.
        """

        self.W = []
        layer_sizes = [self.il] + list(self.hl) + [self.ol]

        #Gewichte initialisieren
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i] + 1  #+1 für Bias
            output_size = layer_sizes[i + 1]
            if self.akt == "relu" and not self.conjugateGradient: 
                #He-Initalisierung für ReLu-Netze
                w = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)  
            elif self.akt == "sigmoid" or self.conjugateGradient:
                #Xavier-Initalisierung für Sigmoid und CG
                w = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)  #Xavier-Init
            self.W.append(w)

        #Conjugate Gradient - Vorbereitung zusätzlicher Strukturen
        if self.conjugateGradient:
            self.gradtmo = [np.zeros_like(w) for w in self.W] #Gradient t-1
            self.steptmo = [np.zeros_like(w) for w in self.W] #Suchrichtung t-1
            self.gradflatprev = np.zeros_like(self._flatten_weights())
            self.stepflatprev = np.zeros_like(self.gradflatprev)


    def _addBias(self, X):
        """
        Fügt der Eingabematrix X eine zusätzliche Spalte für den Bias-Term hinzu.
        """
        return np.hstack((X, np.ones((X.shape[0], 1))))


    def _flatten_weights(self):
        """
        Fasst alle Gewichtsmatrizen der Schichten zu einem einzelnen 1D-Vektor zusammen.
        Wird für Conjugate Gradient benötigt.
        """
        return np.concatenate([w.ravel() for w in self.W])


    def _unflatten_weights(self, weightflat):
        """
        Wandelt einen flachen Gewichtsvektor wieder in die ursprünglichen Schichtmatrizen um.
        """

        new_Weight = []
        index = 0
        for w in self.W:
            size = w.size
            new_W = weightflat[index:index + size].reshape(w.shape)
            new_Weight.append(new_W)
            index += size
        self.W = new_Weight


    def _flatten_grads(self, listgrad):
        """
        Fasst eine Liste von Gradientenmatrizen zu einem einzigen Vektor zusammen.
        Wird bei der Conjugate-Gradient-Methode verwendet, um
        alle Gradienten gemeinsam zu verarbeiten.
        """ 
        return np.concatenate([g.ravel() for g in listgrad])


    def _huber_loss(self, y_true, y_pred):
        """
        Berechnet den Huber-Verlust zwischen den echten Werten y_true und den vorhergesagten Werten y_pred.
        """
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.huberdelta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.huberdelta * (np.abs(error) - 0.5 * self.huberdelta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))


    def _calOut(self, X, biasadded=False, training=True):
        """
        Führt den Vorwärtsdurchlauf (Forward Pass) durch das Netzwerk aus.
        """
        I = X if biasadded else self._addBias(X)
        for i in range(len(self.W) - 1):
            z = self.W[i] @ I.T
            a = self._aktivierung(z).T
            #Dropout nur im Training und in den versteckten Schichten anwenden
            if training and self.dropout > 0 and i < len(self.W) - 2:
                mask = (np.random.rand(*a.shape) > self.dropout).astype(float) / (1 - self.dropout)
                a *= mask
            I = self._addBias(a)
        return (self.W[-1] @ I.T).T


    def predict(self, X):
        """
        Berechnet Modellvorhersagen für Eingabedaten ohne Dropout.
        """
        y = self._calOut(X, training=False)
        return y


    def fit(self, X, Y):
        """
        Trainiert das MLP-Modell mit gegebenen Eingabe- und Zielwerten.
        Führt eine automatische Aufteilung in Trainings- und Validierungsdaten durch
        und ruft intern die Trainingsroutine (_train) auf.
        """
        if Y.ndim == 1:
            Y = Y[:, None] #Sicherstellen, dass Y 2D ist

        self.il = X.shape[1]
        self.ol = Y.shape[1]
        self._initWeights()

        #Aufteilung in Trainings- und Validierungsset
        XVal, YVal, XTrain, YTrain = self._divValTrainSet(X, Y)
        self._train(XTrain, YTrain, XVal, YVal)


    def _train(self, X, Y, XVal=None, YVal=None):
        """
        Interne Trainingsroutine des MLP-Modells.
        Führt den Lernprozess über mehrere Epochen durch, inklusive:
        - Mini-Batch-Training
        - Forward- und Backpropagation
        - Optimizer-Logik (Adam, RMSProp, etc.)
        - Regularisierung
        - Validierungsüberwachung (early stopping)
        """
        X = self._addBias(X) #Bias hinzufügen
        nsamples = X.shape[0] #Anzahl Samples
        counter = 0 #Counter für early stopping
        epoch = 0 #Epoche Zähler
        minError = float('inf') 
        minW = copy.deepcopy(self.W)


        eta = self.eta
        starteta = self.eta
        iterations = 0


        #Optimierungsspezifische Variablen
        self.velocity = [np.zeros_like(w) for w in self.W] #v (Geschwindigkeit)
        av = [np.zeros_like(w) for w in self.W] #accumulation variable
        firstmomentvar = [np.zeros_like(w) for w in self.W]
        secmomentvar = [np.zeros_like(w) for w in self.W]
        firstmomentbias = [np.zeros_like(w) for w in self.W]
        secmomentbias = [np.zeros_like(w) for w in self.W]
        timestep = 0
        alpha = 0

        #Batch Grösse
        echte_batchsize = self.batchsize
        if echte_batchsize == 0 or self.conjugateGradient: #Full batch
            echte_batchsize = X.shape[0]
            print("using full batch")


        #Early stopping loop
        while iterations < self.maxIter:  
            epoch += 1
            mixSet = np.random.permutation(X.shape[0]) #Daten mischen

            #Optimierung mit Mini-Batches
            for batchstart in range(0, nsamples, echte_batchsize):
                batchindex = mixSet[batchstart:batchstart + echte_batchsize] #Batch index
                xbatch = X[batchindex].T
                ybatch = Y[batchindex].T
                
                #Unterteilung in Aktivierungen und Voraktivierungen
                akt = [xbatch] 
                vorakt = []

                if self.Adam: 
                    timestep += 1


                #--- Forward Pass ---
                for i, w in enumerate(self.W):
                    z = w @ akt[-1]
                    vorakt.append(z)
                    if i == len(self.W) - 1:
                        akt.append(z) #Output layer keine Aktivierungsfunktion
                    else:
                        a = self._aktivierung(z) #Hidden layer mit Aktivierungsfunktion
                        #Dropout
                        if self.dropout > 0: 
                            mask = (np.random.rand(*a.shape) > self.dropout).astype(float) / (1 - self.dropout)
                            a = a*mask     
                            self.dropoutused=True  
                        a = np.vstack((a, np.ones((1, a.shape[1]))))
                        akt.append(a)


                #--- Backpropagation ---
                #Fehler berechnen
                if self.lossfunction == "MSE": #Mean Squared Error
                    gradients = [akt[-1] - ybatch]
                elif self.lossfunction == "Huberloss": #Huber Loss
                    error = akt[-1] - ybatch
                    gradients = [
                        np.where(
                            np.abs(error) <= self.huberdelta,
                            error,
                            self.huberdelta * np.sign(error)
                        ) / echte_batchsize
                    ]

                #Gradienten für jede Schicht berechnen
                for l in reversed(range(1, len(self.W))): #Rekursive Berechnung der Gradienten
                    W_no_bias = self.W[l][:, :-1] #Bias Gewichte entfernen
                    delta = (W_no_bias.T @ gradients[0]) * self._ableit(vorakt[l - 1]) #Delta berechnen
                    gradients.insert(0, delta) #Gradientenliste auffüllen
                        
                
                #--- Weight updates ---
                for l in range(len(self.W)): 
                    grad = gradients[l] @ akt[l].T / echte_batchsize #Gradient für Gewichtsmatrix l

                    #Regularisierung anwenden
                    reg = np.copy(self.W[l])
                    reg[:, -1] = 0  #Bias-Gewichte nicht regularisieren
                    #Anwendung der Parameternormstrafen
                    if self.regul == "l2":
                        grad += self.regulrate * reg
                    elif self.regul == "l1":
                        grad += self.regulrate *np.sign(reg)

                    #SDG Update
                    if self.sgdupdate and iterations < self.sgdmaxiterations:
                        alpha = iterations/self.sgdmaxiterations
                        eta = (1-alpha)*starteta + alpha*(eta/100) 
        

                    #---Optimierer-Spezifische Updates---

                    #RMSProp mit Nestrov-Momentum
                    if self.RMSProp and self.nestrov: 
                        vorvelocity = np.copy(self.velocity[l])
                        av[l] = self.RMSProprate*av[l] + (1-self.RMSProprate) * grad**2 #RMSProp: Exp. Mittelwert der Gradientenquadrate (adaptive Stepgröße)
                        self.velocity[l] = self.momentum * self.velocity[l] - eta/(np.sqrt(av[l])+ self.RMSPropconst) * grad #Berechnung des Momentums
                        self.W[l] += -self.momentum * vorvelocity + (1 + self.momentum) * self.velocity[l] #Gewichtsupdate mit Nestrov
                        self.modelused="RMSProp/Nestrov"

                    #Adam 
                    elif self.Adam: 
                        firstmomentvar[l] = self.firstmomentrate*firstmomentvar[l] + (1-self.firstmomentrate) * grad #Exponentiell gewichteter Mittelwert 1. Moment
                        secmomentvar[l] = self.secmomentrate*secmomentvar[l] + (1-self.secmomentrate) * grad ** 2 #Exponentiell gewichteter Mittelwert 2. Moment
                        firstmomentbias[l] = firstmomentvar[l]/(1-self.firstmomentrate**timestep) #Bias Korrektur 1. Moment
                        secmomentbias[l] = secmomentvar[l]/(1-self.secmomentrate**timestep) #Bias Korrektur 2. Moment
                        self.W[l] -= eta*firstmomentbias[l]/(np.sqrt(secmomentbias[l])+self.Adamconst) #Gewichtsupdate
                        self.modelused="Adam"

                    #AdaGrad
                    elif self.adagrad: 
                        av[l] += grad ** 2 #Akkumulation der Gradientenquadrate für adaptive Lernrate
                        adaptivelr = eta / (np.sqrt(av[l]) + self.adagradconst) #Lernenrate pro Gewicht anpassen
                        self.W[l] -= adaptivelr * grad #Gewichtsupdate
                        self.modelused="Adagrad"
                     
                    #RMSProp (ohne Nestrov)
                    elif self.RMSProp: 
                        av[l] = self.RMSProprate*av[l] + (1-self.RMSProprate) * grad**2
                        adaptivelr = eta/(np.sqrt(av[l]+self.RMSPropconst)) #Lernrate adaptieren (moving average)
                        self.W[l] -= adaptivelr * grad #Update mit gleitender Mittelwert-Lernrate
                        self.modelused="RMSProp"

                    #Momentum / Nesterov / SGD
                    elif self.sgdupdate:
                        grad = np.clip(grad, -1.0, 1.0)  # Gradient clipping zur Stabilisierung
                        if self.momentum > 0:
                            if self.nestrov: 
                                vorvelocity = np.copy(self.velocity[l])
                                self.velocity[l] = self.momentum * self.velocity[l] - eta * grad #Vorblick-Speed berechnen
                                self.W[l] += -self.momentum * vorvelocity + (1 + self.momentum) * self.velocity[l] #Gewichtsupdate mit Nestrov
                                self.modelused="Nestrov Momentum"
                            else: 
                                self.velocity[l] = self.momentum * self.velocity[l] - eta * grad #Berechnung des Momentums
                                self.W[l] += self.velocity[l] #Gewichtsupdate mit Momentum
                                self.modelused="Momentum"
                        else: 
                            self.W[l] -= eta * grad #Standard SGD Update
                            self.modelused="SGDUpdate"

                    #---Conjugate Gradient---
                    elif self.conjugateGradient:
                        #Gradienten mit Regularisierung berechnen
                        allgradients = []
                        
                        for indexlayer in range(len(self.W)):
                            # Gradienten für aktuelle Schicht: delta * Aktivierungen der vorherigen Schicht
                            gradientlayer = gradients[indexlayer] @ akt[indexlayer].T / echte_batchsize
                            
                            #Kopie der Gewichte für Regularisierung
                            regw = np.copy(self.W[indexlayer])
                            regw[:, -1] = 0  # Bias-Gewichte nicht regularisieren

                            #Normstrafen anwenden
                            if self.regul == "l2":
                                gradientlayer += self.regulrate * regw
                            elif self.regul == "l1":
                                gradientlayer += self.regulrate * np.sign(regw)
                            
                            # Gradienten der Schicht zur Liste hinzufügen
                            allgradients.append(gradientlayer)

                        # Alle Gradienten zu einem flachen Vektor zusammenfassen
                        gradientnow = self._flatten_grads(allgradients).astype(float)

                        # Gradient-Clipping zur Stabilisierung
                        gradientnorm = np.linalg.norm(gradientnow) #Norm des Gradientenvektors berechnen                                              
                        if gradientnorm > 100:
                            # Wenn Gradient zu groß, auf Norm 100 skalieren
                            gradientnow = gradientnow / gradientnorm * 100

                        # Berechnung der Suchrichtung
                        if iterations == 0 or np.all(self.gradflatprev == 0):
                            # Erste Iteration: Richtung = negativer Gradient
                            stepnow = -gradientnow
                        else:
                            # Polak-Ribière Formel für CG-Richtung
                            gradientdifference = gradientnow - self.gradflatprev #Unterschied zu vorherigem Gradienten
                            denom = np.dot(self.gradflatprev, self.gradflatprev) + 1e-12 #Division durch 0 verhindern
                            betapr = np.dot(gradientnow, gradientdifference) / denom #Beta nach Polak-Ribière
                            betapr = max(betapr, 0.0)  #Beta darf nicht negativ sein
                            stepnow = -gradientnow + betapr * self.stepflatprev #Neue CG-Richtung

                        #Sicherstellen, dass die Richtung absteigend ist (dotprod < 0)
                        dotprod = np.dot(gradientnow, stepnow)
                        if dotprod >= 0:
                            #Falls nicht, zurück auf negativer Gradient
                            stepnow = -gradientnow  
                            dotprod = np.dot(gradientnow, stepnow)

                        #Alte Gewichte zwischenspeichern
                        oldweights = self._flatten_weights().copy()
                        tryalpha = 1.0  #Startwert für Line Search Schrittweite
                        success = False #Erfolgsflag für Line Search

                        #Funktion für Verlustberechnung mit Regularisierung
                        def compute_loss_with_reg(Xb, Yb):
                            #Vorwärtsberechnung
                            yp = self._calOut(Xb, biasadded=True, training=False)
                            mse = np.mean((Yb - yp) ** 2)
                            regulterm = 0
                            for w in self.W:
                                regulw = w.copy()
                                regulw[:, -1] = 0
                                if self.regul == "l2":
                                    regulterm += self.regulrate * np.sum(regulw ** 2)
                                elif self.regul == "l1":
                                    regulterm += self.regulrate * np.sum(np.abs(regulw))
                            return mse + regulterm #Gesamtverlust

                        currentloss = compute_loss_with_reg(X[batchindex], Y[batchindex])

                        #Line Search zur Bestimmung der optimalen Schrittweite
                        for _ in range(self.maxlinesearchiters):
                            #Gewichte für aktuellen Versuch berechnen
                            trialweights = oldweights + tryalpha * stepnow
                            self._unflatten_weights(trialweights) #In Schichten zurückverwandeln

                            trialloss = compute_loss_with_reg(X[batchindex], Y[batchindex])

                            # Armijo-Bedingung prüfen: Verlust muss ausreichend sinken
                            if trialloss <= currentloss + self.c1 * tryalpha * dotprod:
                                
                                # Compute trial gradient for Wolfe
                                trialakt = [X[batchindex].T]
                                vortrialakt = []
                                for wi, currentweight in enumerate(self.W):
                                    ztemp = currentweight @ trialakt[-1]
                                    vortrialakt.append(ztemp)
                                    if wi == len(self.W) - 1:
                                        trialakt.append(ztemp)
                                    else:
                                        tempakt = self._aktivierung(ztemp)
                                        tempakt = np.vstack((tempakt, np.ones((1, tempakt.shape[1]))))
                                        trialakt.append(tempakt)

                                #Rückwärtsberechnung der Gradienten
                                tempgradients = [trialakt[-1] - Y[batchindex].T] #Delta Output Layer
                                for lv in reversed(range(1, len(self.W))):
                                    weightnobias = self.W[lv][:, :-1]
                                    deltanow = (weightnobias.T @ tempgradients[0]) * self._ableit(vortrialakt[lv - 1])
                                    tempgradients.insert(0, deltanow) 

                                #Gradienten für alle Schichten berechnen inkl. Regularisierung
                                gradslisttrial = []
                                for li in range(len(self.W)):
                                    gtemp = tempgradients[li] @ trialakt[li].T / echte_batchsize
                                    regwt = np.copy(self.W[li])
                                    regwt[:, -1] = 0
                                    if self.regul == "l2":
                                        gtemp += self.regulrate * regwt
                                    elif self.regul == "l1":
                                        gtemp += self.regulrate * np.sign(regwt)
                                    gradslisttrial.append(gtemp)
                                
                                gradnowtrial = self._flatten_grads(gradslisttrial)

                                # Strong Wolfe: Richtungsableitung prüfen
                                dderivtrial = np.dot(gradnowtrial, stepnow)
                                if abs(dderivtrial) <= self.c2 * abs(dotprod):
                                    #Schritt erflgreich
                                    success = True
                                    break

                            #Schrittweite verkleinern
                            tryalpha *= self.rho
                            self._unflatten_weights(oldweights)  #Gewichte zurücksetzen

                        #Update der vorherigen Gradienten und Suchrichtungen
                        if success:
                            self.gradflatprev = gradientnow.copy() #Speichern für nächste CG-Iteration
                            self.stepflatprev = stepnow.copy()
                        else:
                            # Falls Line Search fehlschlägt, zurücksetzen
                            self._unflatten_weights(oldweights)  

                        self.modelused = "ConjugateGradient"
                        
                        print(f"Iter {iterations}, Loss: {currentloss:.6f}, Grad norm: {gradientnorm:.6f}, Alpha: {tryalpha:.6f}")

                    else:
                        self.W[l] -= eta*grad
                        self.modelused="N/A"


                iterations += 1                    


                #---Validierung und Fehlerüberwachung---
                yp = self._calOut(XVal)
                if self.lossfunction == "MSE":
                    self.meanEvar = np.mean((YVal - yp) ** 2)
                elif self.lossfunction == "Huberloss":
                    self.meanEvar = self._huber_loss(YVal, yp)

                #Verlauf speichern
                self.YVal = YVal
                self.yp = yp
                self.epoch.append(iterations)
                self.meanE.append(self.meanEvar)

                #Bester Validierungsfehler speichern
                if self.meanEvar < minError:
                    minError = self.meanEvar
                    minW = copy.deepcopy(self.W)
                    self.minError = minError
                    counter = 0
                else:
                    counter += 1
                    #Frühabbruch bei Early Stopping
                    if counter >= self.patience:
                        self.earlystop = True
                        break
            
            #Frühabbruch bei Early Stopping
            if self.earlystop:
                break



        #Beste Gewichte wiederherstellen
        self.W = minW
        self.trainingsteps = iterations


    def _divValTrainSet(self, X, Y):
        """
        Teilt die Eingabedaten in Trainings- und Validierungsdaten auf.
        """
        valmenge = np.random.choice(X.shape[0], int(X.shape[0] * 0.20), replace=False)
        trainmenge = np.setdiff1d(np.arange(X.shape[0]), valmenge)
        
        #Speichere Indexmenge für spätere Verwendung
        self.TrainSet = trainmenge
        self.ValSet = valmenge

        return X[valmenge], Y[valmenge], X[trainmenge], Y[trainmenge]
