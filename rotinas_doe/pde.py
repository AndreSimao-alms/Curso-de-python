# Função -> aghata_efeito (fabi_efeito)

#Função adaptada do "fabi_efeito" utilizada no Octave pertencente ao Prof.Dr.Edenir Pereira Filho para o Python**\
#Canal do Youtube: https://www.youtube.com/c/EdenirPereiraFilho

#Funcao para calcular efeito de planejamento fatorial\
#X = matriz contendo os efeitos que serão calculados\
#y = vetor contendo a resposta\
#erro_efeito=erro de um efeito. Sera 0 se nao forem feitas replicas\
#t=valor de t correspondente ao número de graus de liberdade do erro de um efeito. Sera 0 se nao forem feitas replicas.
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


class Fabi_efeito:
    
    """
    Classe -> Fabi_efeito(X,y,erro_efeito,t)
    
    Instancie esta classe para acessar os seguintes métodos: grafico_probabilidades(), porcentagem_efeitos(), fabi_efeito().
    
    Rotina adaptada do "fabi_efeito" utilizada no Octave pertencente ao Prof.Dr.Edenir Pereira Filho para o Python.
    Canal do Youtube: https://www.youtube.com/c/EdenirPereiraFilho
    
    Parameters
    -----------
    
    X: matriz contendo os efeitos que serão calculados.
    y: vetor contendo a resposta.
    erro_efeito: erro de um efeito. Sera 0 se nao forem feitas replicas.
    t: valor t de distribuição t_Student.
    
    Methods
    -----------
    
    fabi_efeito: retorna gráficos de "Probabilidade" e "Porcentagens Efeitos", tabelas excel com dados gerados.
    
    
    """
    
    
    def __init__(self, x, y, erro_efeito, t):
        self.X = x
        self.y = y
        self.erro_efeito = erro_efeito
        self.t = t
        self.inicio = [0]
        self.centro = []
        self.fim = []
        self.gauss = []

        
    @property
    def __matrix_x(self):
        return self.X

    
    @property
    def vetor_y(self):
        return self.y

    
    @property
    def efeito(self):  # Retorna valores do produto entre efeitos e resposta
        return (self.X.T * self.y).T

    
    @property
    def __n_efeito(self):  # Retorna dimensões da matriz com os efeitos (valor_codificado*resposta)
        return self.X.shape

    
    @property
    def __indice_efeitos(self):  # Retorna lista com respectivas interações
        return self.X.T.index

    
    @property
    def __gerar_inicio_centro_fim_gauss(self):  # Retorna os valosres da gaussiana
        for i in range(self.__n_efeito[1]):
            self.fim.append(self.inicio[i] + (1 / self.__n_efeito[1]))
            self.inicio.append(self.fim[i])
            self.centro.append((self.inicio[i] + self.fim[i]) / 2)
            self.gauss.append(norm.ppf(self.centro))
        return self.gauss

    
    def __calcular_efeitos(self):  # Retorna vetor com efeitos
        return (np.einsum('ij->j', self.efeito)) / (self.__n_efeito[0] / 2)  # np.einsum -> função que soma
        # colunas de uma matriz

        
    def __calcular_porcentagem_efeitos(self):  # Retorna vetor com probabilidade
        return (self.__calcular_efeitos() ** 2 / np.sum(self.__calcular_efeitos() ** 2)) * 100

    
    def __definir_gaussiana(self):  # Retorna os valosres da gaussiana
        return self.__gerar_inicio_centro_fim_gauss[self.__n_efeito[1] - 1]


    def __etiqueta(self,axs):  # Demarca os pontos no gráfico de probabilidades
        for i, label in enumerate(self.__sort_efeitos_probabilidades().index):
            axs[0].annotate(label, (self.__sort_efeitos_probabilidades()['Efeitos'].values[i],
                                 self.__definir_gaussiana()[i]))

    def __sort_efeitos_probabilidades(self):  # Retorna dataframe ordenado de maneira crescente com valores de efeitos
        data = pd.DataFrame({'Efeitos': self.__calcular_efeitos()}, index=self.__indice_efeitos)
        data = data.sort_values('Efeitos', ascending=True)
        return data

    
    def __definir_ic(self):  # Retorna conjunto de pontos do IC
        return np.full(len(self.__definir_gaussiana()), self.erro_efeito * self.t)

    
    def __verificar_ic(self,axs):
        if self.erro_efeito == 0 or self.t == 0:
            pass
        else:
            return self.__plotar_ic(axs)

    def __graficos_fabi_efeito(self):
        fig, axs =plt.subplots(2,1,figsize=(6,8))
        
        axs[0].scatter(self.__sort_efeitos_probabilidades()['Efeitos'],
                    self.__definir_gaussiana(), s=40, color='darkred')
        axs[0].set_title('Gráfico de Probabilidades', fontsize=18, fontweight='black', loc='left')
        axs[0].set_ylabel('z')
        axs[0].set_xlabel('Efeitos')  
        self.__etiqueta(axs)
        self.__verificar_ic(axs) 
        axs[0].grid(color='k', linestyle='solid')
        
        sns.set_style("whitegrid")
        sns.load_dataset("tips")
        
        axs[1] = sns.barplot(x='Efeitos', y='%', color='purple', data=pd.DataFrame(
            {'Efeitos': self.__indice_efeitos, '%': self.__calcular_porcentagem_efeitos()}))
        axs[1].set_title('Porcentagem Efeitos', fontsize=16, fontweight='black', loc='left')
        
        fig.suptitle('Gráficos Fabi Efeito', fontsize=22, y=0.99, fontweight='black',color='darkred')
        plt.tight_layout()
        plt.savefig('graficos_fabi_efeito.pdf')
        
    def __plotar_ic(self,axs):  
        axs[0].plot(-self.__definir_ic(), self.__definir_gaussiana(), color='red')
        axs[0].plot(0 * self.__definir_ic(), self.__definir_gaussiana(), color='blue')
        axs[0].plot(self.__definir_ic(), self.__definir_gaussiana(), color='red')

    def fabi_efeito(self):
        """
        Função -> aghata_efeito (fabi_efeito)
        Funcao para calcular efeito de planejamento fatorial
        
        Função adaptada do "fabi_efeito" utilizada no Octave pertencente ao Prof.Dr.Edenir Pereira Filho para o Python**
        Canal do Youtube: https://www.youtube.com/c/EdenirPereiraFilho
        
        Parameters
        -----------
        
        X = matriz contendo os efeitos que serão calculados
        y = vetor contendo a resposta
        erro_efeito=erro de um efeito. Sera 0 se nao forem feitas replicas
        t=valor de t correspondente ao número de graus de liberdade do erro de um efeito. Sera 0 se nao forem feitas replicas.
        
        Returns
        -----------
        
        Gráficos de "Porcentagem x efeitos" (barplot) e "Probabilidade" (scatter) oriundo da rotina fabi_efeito do Octave
        
        
        """
        return plt.show(self.__graficos_fabi_efeito())

class CP:
    """
    Classe CP: responsável por calcular valor t e erro de um efeito
    
    
    Parameters
    -----------
    
    y: pd.Series - valores dos sinais da região do ponto central.
    
    k: int -  número de variáveis. 
    
    Methods
    -----------
    
    invt: retorna t-value.
    
    erro_efeito: retorna erro de um efeito.
    
    Return
    -----------
    
    tuple -> (erro_efeito, t)
    
    
    """
    def __init__(self,y, k):
        self.y = y
        self.k = k
        
    def __array(self):
        return self.y.values
    
    def __erro_exp(self):
        return self.y.std()
    
    def __gl(self):
        return self.y.shape[0]-1
    
    def invt(self):
        return t.ppf(1-.05/2,self.__gl())
    
    def erro_efeito(self):
        return 2*self.__erro_exp()/(self.y.shape[0]*2**self.k)**0.5
    
