# MAIN
setup="online"

# importa il pacchetto json 
import json
# importa la classe Ottimizzatore
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
folder_ottimizzatore_path = os.path.join(current_dir, 'EMS')
sys.path.append(folder_ottimizzatore_path)
from file_classe_EMS import EMS
from file_ottimizza_FIS import ottimizzaFIS
# importa la classe della microgrid
from MG.file_classe_microgrid import MG
# importa il modulo per generare numeri random
import numpy as np
# np.random.seed(1331)
# importa il modulo per plottare
import matplotlib.pyplot as plt
# importa il modulo per calcoalre il fis ottimizzato
from FO.globale_fine_ottimizzazione import globale_fine_ottimizzazione
import time
import matplotlib.pyplot as plt
import math
import statistics


# Start timer
start_time = time.time()


# SIMULAZIONE INTERNA PER AUTOCONSUMO

# parametri comuni a tutte le microgrid
PR_3=150  # prezzo dell'energia per il ritiro dedicato [€/MWh]
TRAS_e=8.48  # tariffa di trasmissione definita per le utenze in bassa tensione [€/MWh]
max_BTAU_m=0.61  # valore maggiore della componente variabile di distribuzione [€/MWh]
CPR=0.026  # coefficiente delle perdite di rete evitate [-]
Pz=3.2  # prezzo zonale orario [€/MWh]
a_x=694 #(2800)
b_x=0.795
B_x=4500



# microgrid 1
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.50
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=7400 # costo installazione PV [€]
MG_1=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 2
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.50
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=6000 # costo installazione PV [€]
MG_2=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 3
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.50
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=7400 # costo installazione PV [€]
MG_3=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 4
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.50
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=7400 # costo installazione PV [€]
MG_4=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 5
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.50
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=6000 # costo installazione PV [€]
MG_5=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 6
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.50
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteria
pv_price=7400 # costo installazione PV [€]
MG_6=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)

# microgrid 7
is_CER=1
TP_CE=110 # tariffa premio [€/MWh]: 110, se si tratta di una CER; 100, se si tratta di un AUC
SoC_0=0.50
eta=0.98  # rendimento della batteria
SoC_min=0.15  # Stato di Carica minimo della batteria 
SoC_max=0.95  # Stato di Carica massimo della batteria
Q=5  # Capacità della batteria [kWh]
P_S_max=7  # massima Potenza di carica/scarica della batteria [kW]
B=B_x  # prezzo della batteria [€]
a=a_x  # parametro della curva cicli/DoD della batteria
b=b_x  # parametro della curva cicli/DoD della batteriab=1.665  # parametro della curva cicli/DoD della batteria
pv_price=6000 # costo installazione PV [€]
MG_7=MG(is_CER, SoC_0, Q, P_S_max,eta, SoC_min,SoC_max,PR_3,B,a,b,CPR,
             Pz,TP_CE, TRAS_e, max_BTAU_m,SoC_0,pv_price,SoC_0)



# LEGGE I METADATI PER L'OTTIMIZZAZIONE
with open("meta-dati.json") as file:
    file_meta_dati=json.load(file)
nome_funzione_obiettivo=file_meta_dati["FO"]
limiti_variabili=[]
metaparametri=[]
funzione_obiettivo=None
numero_esecuzioni=file_meta_dati["numero_esecuzioni"]
if nome_funzione_obiettivo=="globale":
    from FO.globale import globale
    funzione_obiettivo=globale
    limiti_variabili=file_meta_dati["limiti_variabili_globale"]
if nome_funzione_obiettivo=="modello_CER_AUC":
    from FO.modello_CER_AUC import modello_CER_AUC
    funzione_obiettivo=modello_CER_AUC
    limiti_variabili=file_meta_dati["limiti_variabili_CER_AUC"]
if nome_funzione_obiettivo=="rastrigin":
    from FO.rastrigin import rastrigin
    funzione_obiettivo=rastrigin
    limiti_variabili=file_meta_dati["limiti_variabili_rastrigin"]
    algoritmo_di_ottimizzazione=file_meta_dati["algoritmo_di_ottimizzazione"]
if nome_funzione_obiettivo=="rosenbrock":
    from FO.rosenbrock import rosenbrock
    funzione_obiettivo=rosenbrock
    limiti_variabili=file_meta_dati["limiti_variabili_rosenbrock"]
    algoritmo_di_ottimizzazione=file_meta_dati["algoritmo_di_ottimizzazione"]
if nome_funzione_obiettivo=="sferica":
    from FO.sferica import sferica
    funzione_obiettivo=sferica
    limiti_variabili=file_meta_dati["limiti_variabili_sferica"]
    algoritmo_di_ottimizzazione=file_meta_dati["algoritmo_di_ottimizzazione"]
if nome_funzione_obiettivo=="schwefel":
    from FO.schwefel import schwefel
    funzione_obiettivo=schwefel
    limiti_variabili=file_meta_dati["limiti_variabili_schwefel"]
    algoritmo_di_ottimizzazione=file_meta_dati["algoritmo_di_ottimizzazione"]
if nome_funzione_obiettivo=="griewank":
    from FO.griewank import griewank
    funzione_obiettivo=griewank
    limiti_variabili=file_meta_dati["limiti_variabili_griewank"]
algoritmo_di_ottimizzazione=file_meta_dati["algoritmo_di_ottimizzazione"]
if algoritmo_di_ottimizzazione=="GA":
    metaparametri=file_meta_dati["metaparametri_GA"]
else:
    print("Non ci sono altri algoritmi implementati!")



# OTTIMIZZAZIONE DEL FIS

# rolling-time horizon
if setup=="rolling":
    # carica l'intero training set
    import scipy.io
    training_set_matlab = scipy.io.loadmat('training_set_per_python.mat')
    training_set = training_set_matlab['training_set_per_python']
    lunghezza_training_set=48
    training_set=training_set[0:14,288:288+lunghezza_training_set]
    ###
    # per test online
    # test_set_matlab = scipy.io.loadmat('test_set_per_python.mat')
    # test_set = test_set_matlab['test_set_per_python']
    # lunghezza_test_set=96
    # training_set=test_set[0:14,0:lunghezza_test_set]
    
    # dati sulle MG (modello fisico simulato)
    dati_MG=[MG_1,
             MG_2,
             MG_3,
             MG_4,
             MG_5,
             MG_6,
             MG_7]
    
    # carica il fis
    dati_fis=[]
    with open("fis.json") as file:
        dati_fis=json.load(file)
        
       
    # ottimizza il fis
    n_esecuzioni=1
    n_geni=30;
    fo_ottime=np.zeros([n_esecuzioni])
    individui_ottimi=np.zeros([n_esecuzioni,n_geni])
    for esecuzione in range(n_esecuzioni):
        print('ESECUZIONE: ',str(esecuzione))
        dati_modello=[dati_fis,dati_MG,training_set]
        risultati_ottimizzazione=ottimizzaFIS(dati_modello,
                                funzione_obiettivo, 
                                limiti_variabili, 
                                metaparametri,
                                nome_funzione_obiettivo, 
                                esecuzione)
        fo_ottima = risultati_ottimizzazione[0][1][0]
        individuo_ottimo =risultati_ottimizzazione[0][2][0]
        fo_ottime[esecuzione]=fo_ottima
        individui_ottimi[esecuzione,:]=individuo_ottimo
    fo_ottima_media = statistics.mean(fo_ottime)
    fo_ottima_varianza = statistics.pvariance(fo_ottime)
    
    individuo_ottimo=np.zeros([n_geni])
    varianze=np.zeros([n_geni])
    for g in range(n_geni):
        somma_geni=0
        somma_numeratori_singoli=0
        for e in range(n_esecuzioni):
            gene = individui_ottimi[e][g]
            somma_geni= somma_geni+gene
        media=somma_geni/n_esecuzioni
        for es in range(n_esecuzioni): 
            gene = individui_ottimi[es][g]
            numeratore_singolo=(gene-media)**2
            somma_numeratori_singoli=somma_numeratori_singoli+numeratore_singolo
        varianza=somma_numeratori_singoli/n_esecuzioni
        individuo_ottimo[g]=media
        varianze[g]=varianza
    
    # End timer
    end_time = time.time()
    # Calcola il tempo di esecuzione
    elapsed_time_training = end_time - start_time
    
    # test 
    test_set_matlab = scipy.io.loadmat('test_set_per_python.mat')
    test_set = test_set_matlab['test_set_per_python']
    lunghezza_test_set=96
    test_set=test_set[0:14,0:lunghezza_test_set]
    dati_modello_test=[dati_fis,dati_MG, test_set]
    
    # Start timer
    start_time_test = time.time()
    risultati_test=globale_fine_ottimizzazione(dati_modello_test,individuo_ottimo,"rolling")
    # End timer
    end_time_test = time.time()
    # Calcola il tempo di esecuzione
    elapsed_time_test = end_time_test - start_time_test

    FO_globale_energy_community=risultati_test[0]
    matrice_decisioni=risultati_test[1]
    matrice_SoC=risultati_test[2]
    matrice_P_GL_S=risultati_test[3]
    matrice_P_GL_N=risultati_test[4]
    matrice_FO=risultati_test[5]
    costo_computazionale=risultati_test[6]
    fis_ottimo=risultati_test[7]
    ascisse_MF_input=risultati_test[8]
    ascisse_MF_output=risultati_test[9]
    conseguenti=risultati_test[10]
    pesi_regole=risultati_test[11]
    FO_autoconsumo_energy_community=risultati_test[12]
    vettore_FO=risultati_test[13]
    
    # salva i risultati
    nome_sottocartella = "Risultati"
    cartella_corrente = os.path.dirname(os.path.abspath(__file__))
    percorso_sottocartella = os.path.join(cartella_corrente, nome_sottocartella)
    
    nome_file = "individuo_ottimo.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),individuo_ottimo)
    
    nome_file = "varianze_geni.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),varianze)
    
    nome_file = "n_esecuzioni.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),n_esecuzioni)
    
    nome_file = "fis_ottimo.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),fis_ottimo)
    
    nome_file = "FO_ottima_training_set.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),fo_ottima_media)
    
    nome_file = "FO_ottima_training_set_varianza.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),fo_ottima_varianza)
    
    nome_file = "tempo_training.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),elapsed_time_training)
    
    nome_file = "tempo_test.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),elapsed_time_test)
    
    nome_file = "FO_ottima_test_set.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),FO_globale_energy_community)
    
    nome_file = "matrice_decisioni.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),matrice_decisioni)
    
    nome_file = "matrice_SoC.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),matrice_SoC)
    
    nome_file = "matrice_P_GL_S.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),matrice_P_GL_S)
    
    nome_file = "matrice_P_GL_N.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),matrice_P_GL_N)
    
    nome_file = "matrice_FO.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),matrice_FO)
    
    
     
    
    # resa grafica
    X=np.arange(0,len(test_set[0]),1)
    fig_1 = plt.figure()
    plt.title('Alfa')
    plt.scatter(X,matrice_decisioni[0],1,'k',label="MG 1")
    plt.scatter(X,matrice_decisioni[1],1,'b',label="MG 2")
    plt.scatter(X,matrice_decisioni[2],1,'g',label="MG 3")
    plt.scatter(X,matrice_decisioni[3],1,'r',label="MG 4")
    plt.scatter(X,matrice_decisioni[4],1,'y',label="MG 5")
    plt.scatter(X,matrice_decisioni[5],1,'m',label="MG 6")
    plt.scatter(X,matrice_decisioni[6],1,'c',label="MG 7")
    plt.xlabel('timeslot')           
    plt.ylabel('Alfa')
    plt.legend()
    plt.draw()
    nome_file = 'alfa.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    #
    fig_2 = plt.figure()
    plt.title('SoC')
    plt.plot(X,matrice_SoC[0],1,'k',label="MG 1")
    plt.plot(X,matrice_SoC[1],1,'b',label="MG 2")
    plt.plot(X,matrice_SoC[2],1,'g',label="MG 3")
    plt.plot(X,matrice_SoC[3],1,'r',label="MG 4")
    plt.plot(X,matrice_SoC[4],1,'y',label="MG 5")
    plt.plot(X,matrice_SoC[5],1,'m',label="MG 6")
    plt.plot(X,matrice_SoC[6],1,'c',label="MG 7")
    plt.xlabel('timeslot')           
    plt.ylabel('SoC')
    plt.legend()
    plt.draw()
    nome_file = 'SoC.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    #
    fig_3 = plt.figure()
    plt.title('P GL->S')
    plt.plot(X,matrice_P_GL_S[0],3,'k',label="MG 1")
    plt.plot(X,matrice_P_GL_S[1],3,'b',label="MG 2")
    plt.plot(X,matrice_P_GL_S[2],3,'g',label="MG 3")
    plt.plot(X,matrice_P_GL_S[3],3,'r',label="MG 4")
    plt.plot(X,matrice_P_GL_S[4],3,'y',label="MG 5")
    plt.plot(X,matrice_P_GL_S[5],3,'m',label="MG 6")
    plt.plot(X,matrice_P_GL_S[6],3,'m',label="MG 7")
    plt.xlabel('timeslot')           
    plt.ylabel('P GL->S [kW]')
    plt.legend()
    plt.draw()
    nome_file = 'P_GL_S.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    #
    fig_4 = plt.figure()
    plt.title('P GL->N')
    plt.plot(X,matrice_P_GL_N[0],3,'k',label="MG 1")
    plt.plot(X,matrice_P_GL_N[1],3,'b',label="MG 2")
    plt.plot(X,matrice_P_GL_N[2],3,'g',label="MG 3")
    plt.plot(X,matrice_P_GL_N[3],3,'r',label="MG 4")
    plt.plot(X,matrice_P_GL_N[4],3,'y',label="MG 5")
    plt.plot(X,matrice_P_GL_N[5],3,'m',label="MG 6")
    plt.plot(X,matrice_P_GL_N[6],3,'m',label="MG 7")
    plt.xlabel('timeslot')           
    plt.ylabel('P GL->N [kW]')
    plt.legend()
    plt.draw()
    nome_file = 'P_GL_N.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    
    #
    fig_5 = plt.figure()
    plt.title('FO autoc. vs FO globale [€]: '+str(round(FO_autoconsumo_energy_community,2))+'|'+str(round(FO_globale_energy_community,2)))
    plt.plot(X,matrice_FO[0],1,'k',label="MG 1")
    plt.plot(X,matrice_FO[1],1,'b',label="MG 2")
    plt.plot(X,matrice_FO[2],1,'g',label="MG 3")
    plt.plot(X,matrice_FO[3],1,'r',label="MG 4")
    plt.plot(X,matrice_FO[4],1,'y',label="MG 5")
    plt.plot(X,matrice_FO[5],1,'m',label="MG 6")
    plt.plot(X,matrice_FO[6],1,'c',label="MG 7")
    plt.xlabel('timeslot')           
    plt.ylabel('FO [€]')
    plt.legend()
    plt.draw()
    nome_file = 'FO.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
        
        
    # FIS
    # input term set
    # trap
    points = [(ascisse_MF_input[0][0],0),(ascisse_MF_input[0][1],1),(ascisse_MF_input[0][2],1),
              (ascisse_MF_input[0][3],0)]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label='Very Low', color='red')
    #tri
    points = [(ascisse_MF_input[1][0],0),(ascisse_MF_input[1][1],1),(ascisse_MF_input[1][2],0)]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label='Low', color='Orange')
    #tri
    points = [(ascisse_MF_input[2][0],0),(ascisse_MF_input[2][1],1),(ascisse_MF_input[2][2],0)]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label='Medium', color='Cyan')
    #tri
    points = [(ascisse_MF_input[3][0],0),(ascisse_MF_input[3][1],1),(ascisse_MF_input[3][2],0)]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label='High', color='Blue')
    # trap
    points = [(ascisse_MF_input[4][0],0),(ascisse_MF_input[4][1],1),(ascisse_MF_input[4][2],1),
              (ascisse_MF_input[4][3],0)]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label='Very High', color='Purple')
    plt.title('Input Term Set')
    plt.xlabel('x')
    plt.ylabel('m(x)')
    plt.legend()
    plt.draw()
    nome_file = 'input_term_set.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    
    # output term set
    # trap
    points = [(ascisse_MF_output[0][0],0),(ascisse_MF_output[0][1],1),(ascisse_MF_output[0][2],1),
              (ascisse_MF_output[0][3],0)]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label='Very Low', color='red')
    #tri
    points = [(ascisse_MF_output[1][0],0),(ascisse_MF_output[1][1],1),(ascisse_MF_output[1][2],0)]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label='Low', color='Orange')
    #tri
    points = [(ascisse_MF_output[2][0],0),(ascisse_MF_output[2][1],1),(ascisse_MF_output[2][2],0)]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label='Medium', color='Cyan')
    #tri
    points = [(ascisse_MF_output[3][0],0),(ascisse_MF_output[3][1],1),(ascisse_MF_output[3][2],0)]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label='High', color='Blue')
    # trap
    points = [(ascisse_MF_output[4][0],0),(ascisse_MF_output[4][1],1),(ascisse_MF_output[4][2],1),
              (ascisse_MF_output[4][3],0)]
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label='Very High', color='Purple')
    plt.title('Output Term Set')
    plt.xlabel('x')
    plt.ylabel('m(x)')
    plt.legend()
    plt.draw()
    nome_file = 'output_term_set.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    
    #regole
    nome_file = "conseguenti.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),conseguenti)
    nome_file = "pesi_regole.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),pesi_regole)






# online
if setup=="online":
    ###############
    # dati sulle MG (modello fisico simulato)
    dati_MG=[MG_1,
             MG_2,
             MG_3,
             MG_4,
             MG_5,
             MG_6,
             MG_7]
    
    # carica il fis
    dati_fis=[]
    with open("fis.json") as file:
        dati_fis=json.load(file)
    # individuo ottimo da rolling
    # individuo_ottimo_salvato_rolling=[
    #     0.687594,
    #     0.763351,
    #     0.952734,
    #     1.16169,
    #     0.146971,
    #     1.40783,
    #     0.0622623,
    #     1.3001,
    #     1.82386,
    #     0.33328,
    #     0.0448805,
    #     0.061002,
    #     1.97885,
    #     0.643861,
    #     1.87123,
    #     0.31596,
    #     1.04861,
    #     0.444738,
    #     3.89041,
    #     0.545574,
    #     0.391291,
    #     0.341601,
    #     0.754028,
    #     0.517738,
    #     0.199304,
    #     4.80173,
    #     1.35817,
    #     1.38289,
    #     4.63747,
    #     1.14298,
    #     ]
    
    # individuo_ottimo_salvato_rolling=[
    #     1.96624,
    #     0.262235,
    #     0.968207,
    #     1.02229,
    #     0.14471,
    #     1.56114,
    #     0.132293,
    #     1.4322,
    #     1.58486,
    #     0.067988,
    #     0.317778,
    #     0.0971036,
    #     2.05741,
    #     1.19298,
    #     1.51191,
    #     0.0878775,
    #     1.4729,
    #     0.269245,
    #     3.86711,
    #     0.569035,
    #     0.0680839,
    #     0.902494,
    #     0.859806,
    #     0.927327,
    #     0.648921,
    #     4.89031,
    #     1.44476,
    #     1.6321,
    #     3.35308,
    #     1.26272
    #     ]
    
    individuo_ottimo_salvato_rolling=[
        1.10117,
        0.509253,
        2.33941,
        1.80622,
        1.40238,
        1.06575,
        1.799,
        0.727085,
        2.77718,
        0.975212,
        0.84891,
        0.576847,
        2.40278,
        0.0940253,
        2.70919,
        1.29952,
        2.4814,
        1.68021,
        3.3186,
        0.86323,
        0.172594,
        0.639724,
        0.827852,
        0.834031,
        0.221491,
        3.6487,
        1.71688,
        2.45422,
        1.12732,
        1.13699
        ]

    somma_FO_REC_online=0
    somma_FO_REC_online_auto=0
    lunghezza_simulazione=96
    vettore_tempi_timeslot = [0]*lunghezza_simulazione
    vettore_FO_online=[0]*lunghezza_simulazione
    vettore_FO_online_auto=[0]*lunghezza_simulazione
    matrice_decisioni_online=[[0 for _ in range(96)] for _ in range(7)]
    matrice_SoC_online=[[0 for _ in range(96)] for _ in range(7)]
    matrice_P_GL_S_online=[[0 for _ in range(96)] for _ in range(7)]
    matrice_P_GL_N_online=[[0 for _ in range(96)] for _ in range(7)]
    import scipy.io
    test_set_matlab = scipy.io.loadmat('test_set_per_python.mat')
    test_set = test_set_matlab['test_set_per_python']
    for timeslot_scorso in range(0,lunghezza_simulazione):
        print('TIMESLOT: ', str(timeslot_scorso))
        # test 
        test_set_online=test_set[0:14,timeslot_scorso:timeslot_scorso+1]   
        dati_modello_test=[dati_fis,dati_MG, test_set_online]
        # Start timer
        start_time_test = time.time()
        risultati_test=globale_fine_ottimizzazione(dati_modello_test,individuo_ottimo_salvato_rolling,"online")
        FO_globale_REC_timeslot=risultati_test[0]
        vettore_decisioni=risultati_test[1]
        matrice_decisioni_online[0][timeslot_scorso]=vettore_decisioni[0][0]
        matrice_decisioni_online[1][timeslot_scorso]=vettore_decisioni[1][0]
        matrice_decisioni_online[2][timeslot_scorso]=vettore_decisioni[2][0]
        matrice_decisioni_online[3][timeslot_scorso]=vettore_decisioni[3][0]
        matrice_decisioni_online[4][timeslot_scorso]=vettore_decisioni[4][0]
        matrice_decisioni_online[5][timeslot_scorso]=vettore_decisioni[5][0]
        matrice_decisioni_online[6][timeslot_scorso]=vettore_decisioni[6][0]
        vettore_SoC=risultati_test[2]
        matrice_SoC_online[0][timeslot_scorso]=vettore_SoC[0][0] 
        matrice_SoC_online[1][timeslot_scorso]=vettore_SoC[1][0]
        matrice_SoC_online[2][timeslot_scorso]=vettore_SoC[2][0]
        matrice_SoC_online[3][timeslot_scorso]=vettore_SoC[3][0]
        matrice_SoC_online[4][timeslot_scorso]=vettore_SoC[4][0]
        matrice_SoC_online[5][timeslot_scorso]=vettore_SoC[5][0]
        matrice_SoC_online[6][timeslot_scorso]=vettore_SoC[6][0]
        vettore_P_GL_S=risultati_test[3]
        matrice_P_GL_S_online[0][timeslot_scorso]=vettore_P_GL_S[0][0] 
        matrice_P_GL_S_online[1][timeslot_scorso]=vettore_P_GL_S[1][0]
        matrice_P_GL_S_online[2][timeslot_scorso]=vettore_P_GL_S[2][0]
        matrice_P_GL_S_online[3][timeslot_scorso]=vettore_P_GL_S[3][0]
        matrice_P_GL_S_online[4][timeslot_scorso]=vettore_P_GL_S[4][0]
        matrice_P_GL_S_online[5][timeslot_scorso]=vettore_P_GL_S[5][0]
        matrice_P_GL_S_online[6][timeslot_scorso]=vettore_P_GL_S[6][0]
        vettore_P_GL_N=risultati_test[4]
        matrice_P_GL_N_online[0][timeslot_scorso]=vettore_P_GL_N[0][0] 
        matrice_P_GL_N_online[1][timeslot_scorso]=vettore_P_GL_N[1][0]
        matrice_P_GL_N_online[2][timeslot_scorso]=vettore_P_GL_N[2][0]
        matrice_P_GL_N_online[3][timeslot_scorso]=vettore_P_GL_N[3][0]
        matrice_P_GL_N_online[4][timeslot_scorso]=vettore_P_GL_N[4][0]
        matrice_P_GL_N_online[5][timeslot_scorso]=vettore_P_GL_N[5][0]
        matrice_P_GL_N_online[6][timeslot_scorso]=vettore_P_GL_N[6][0]
        FO_autoconsumo_timeslot=risultati_test[12]
        vettore_FO_online[timeslot_scorso]=FO_globale_REC_timeslot
        vettore_FO_online_auto[timeslot_scorso]=FO_autoconsumo_timeslot
        somma_FO_REC_online=somma_FO_REC_online+FO_globale_REC_timeslot
        somma_FO_REC_online_auto=somma_FO_REC_online_auto+FO_autoconsumo_timeslot
        # End timer
        end_time_test = time.time()
        # Calcola il tempo di esecuzione
        elapsed_time_test = end_time_test - start_time_test 
        vettore_tempi_timeslot[timeslot_scorso]=elapsed_time_test
    tempo_timeslot_medio=np.mean(vettore_tempi_timeslot)
    # resa grafica e salvataggi
    nome_sottocartella = "Risultati"
    cartella_corrente = os.path.dirname(os.path.abspath(__file__))
    percorso_sottocartella = os.path.join(cartella_corrente, nome_sottocartella)
    X=np.arange(0,len(test_set[0]),1)
    fig_1 = plt.figure()
    plt.title('Alfa online')
    plt.scatter(X,matrice_decisioni_online[0],1,'k',label="MG 1")
    plt.scatter(X,matrice_decisioni_online[1],1,'b',label="MG 2")
    plt.scatter(X,matrice_decisioni_online[2],1,'g',label="MG 3")
    plt.scatter(X,matrice_decisioni_online[3],1,'r',label="MG 4")
    plt.scatter(X,matrice_decisioni_online[4],1,'y',label="MG 5")
    plt.scatter(X,matrice_decisioni_online[5],1,'m',label="MG 6")
    plt.scatter(X,matrice_decisioni_online[6],1,'c',label="MG 7")
    plt.xlabel('timeslot')           
    plt.ylabel('Alfa')
    plt.legend()
    plt.draw()
    nome_file = 'alfa_online.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    #
    fig_2 = plt.figure()
    plt.title('SoE online')
    plt.plot(X,matrice_SoC_online[0],1,'k',label="MG 1")
    plt.plot(X,matrice_SoC_online[1],1,'b',label="MG 2")
    plt.plot(X,matrice_SoC_online[2],1,'g',label="MG 3")
    plt.plot(X,matrice_SoC_online[3],1,'r',label="MG 4")
    plt.plot(X,matrice_SoC_online[4],1,'y',label="MG 5")
    plt.plot(X,matrice_SoC_online[5],1,'m',label="MG 6")
    plt.plot(X,matrice_SoC_online[6],1,'c',label="MG 7")
    plt.xlabel('timeslot')           
    plt.ylabel('SoC')
    plt.legend()
    plt.draw()
    nome_file = 'SoE_online.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    #
    fig_3 = plt.figure()
    plt.title('P GL->S online')
    plt.plot(X,matrice_P_GL_S_online[0],3,'k',label="MG 1")
    plt.plot(X,matrice_P_GL_S_online[1],3,'b',label="MG 2")
    plt.plot(X,matrice_P_GL_S_online[2],3,'g',label="MG 3")
    plt.plot(X,matrice_P_GL_S_online[3],3,'r',label="MG 4")
    plt.plot(X,matrice_P_GL_S_online[4],3,'y',label="MG 5")
    plt.plot(X,matrice_P_GL_S_online[5],3,'m',label="MG 6")
    plt.plot(X,matrice_P_GL_S_online[6],3,'m',label="MG 7")
    plt.xlabel('timeslot')           
    plt.ylabel('P GL->S [kW]')
    plt.legend()
    plt.draw()
    nome_file = 'P_GL_S_online.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    #
    fig_4 = plt.figure()
    plt.title('P GL->N online')
    plt.plot(X,matrice_P_GL_N_online[0],3,'k',label="MG 1")
    plt.plot(X,matrice_P_GL_N_online[1],3,'b',label="MG 2")
    plt.plot(X,matrice_P_GL_N_online[2],3,'g',label="MG 3")
    plt.plot(X,matrice_P_GL_N_online[3],3,'r',label="MG 4")
    plt.plot(X,matrice_P_GL_N_online[4],3,'y',label="MG 5")
    plt.plot(X,matrice_P_GL_N_online[5],3,'m',label="MG 6")
    plt.plot(X,matrice_P_GL_N_online[6],3,'m',label="MG 7")
    plt.xlabel('timeslot')           
    plt.ylabel('P GL->N [kW]')
    plt.legend()
    plt.draw()
    nome_file = 'P_GL_N_online.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    #
    fig_5 = plt.figure()
    plt.title('FO online vs FO autoc. [€]: '+str(round(somma_FO_REC_online,2))+'|'+str(round(somma_FO_REC_online_auto,2)))
    plt.plot(X,vettore_FO_online,1,'k',label="FO")
    plt.plot(X,vettore_FO_online_auto,1,'b',label="FO autoc.")
    plt.xlabel('timeslot')           
    plt.ylabel('FO [€]')
    plt.legend()
    plt.draw()
    nome_file = 'FO_vs_auto_online.eps'
    plt.savefig(os.path.join(percorso_sottocartella, nome_file), format='eps')
    plt.show()
    # 
    nome_file = "matrice_decisioni_online.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),matrice_decisioni_online)
    nome_file = "matrice_SoE_online.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),matrice_SoC_online)
    nome_file = "matrice_P_GL_S_online.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),matrice_P_GL_S_online)
    nome_file = "matrice_P_GL_N_online.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),matrice_P_GL_N_online)
    nome_file = "vattore_FO_online.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),vettore_FO_online)
    nome_file = "vattore_FO_online_auto.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),vettore_FO_online_auto)
    nome_file = "FO_online.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),somma_FO_REC_online)
    nome_file = "FO_auto_online.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),somma_FO_REC_online_auto)
    nome_file = "tempo_medio_timeslot.npy"
    np.save(os.path.join(percorso_sottocartella, nome_file),tempo_timeslot_medio)
    ###########################
    


print("fine")










