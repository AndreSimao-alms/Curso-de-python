#class Fabi_efeito 111
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
#class CP
from scipy.stats import t
#class Regression2
from scipy.stats import f
from scipy.stats import linregress
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display, Latex
import sys
#class Super_fabi
#from sympy import symbols,diff,solve,subs 
from sympy import symbols

class Fabi_efeito:
    
    """
    Classe -> Fabi_efeito(X,y,erro_efeito,t) - Classe para calcular efeito de planejamento fatorial.
    
    Instancie esta classe para acessar os seguintes métodos: grafico_probabilidades(), porcentagem_efeitos(), fabi_efeito().
    
    Rotina adaptada do "fabi_efeito" utilizada no Octave pertencente ao Prof.Dr.Edenir Pereira Filho para o Python.
    Canal do Youtube: https://www.youtube.com/c/EdenirPereiraFilho
    
    Atributes
    -----------
    
    X: matriz contendo os efeitos que serão calculados.
    y: vetor contendo a resposta.
    erro_efeito: erro de um efeito. Sera 0 se nao forem feitas replicas.
    t: valor t de distribuição t_Student.
    
    Methods
    -----------
    
    fabi_efeito: retorna gráficos de "Probabilidade" e "Porcentagens Efeitos", tabelas excel com dados gerados.
    
    
    """
    
    
    def __init__(self, x, y, erro_efeito=0, t=0):
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
        Função -> fabi_efeito
        Função para calcular efeito de planejamento fatorial
        
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
    Classe -> CP(y, k) - responsável por calcular valor t e erro de um efeito.
    
    
    Atributes
    -----------
    
    y: pd.Series - valores dos sinais da região do ponto central.
    
    k: int -  número de variáveis. 
    
    Methods
    -----------
    
    invt: retorna t-value.
    
    erro_efeito: retorna erro de um efeito.
    
    SSPE: retorna o valor da Soma Quadrática do Erro Puro
    
    df_SSPE: retorna os graus de liberdade da Soma Quadrática do Erro Puro

    
    """
    def __init__(self,y=None , k=None):
        self.y = y
        self.k = k

        
    def __array(self): 
        return self.y.values
    
    def __erro_exp(self):
        return self.y.std()
    
    def __df(self):
        """Calcula valor de t da distribuição bimodal t-Student"""
        return self.y.shape[0]-1
    
    def __verificar_df(self):
        return 
    
    def invt(self, df_a = None):
        """
        Retorna t-value da distribuição bimodal t_Student.
        
        Parameters
        -----------
        
        (optional) df_a:grau de liberdade que não pertence à classe CP.
        
        Returns:
        
        t-value type float
        
        """
        if (df_a == None):
            return t.ppf(1-.05/2,self.__df())
        else:
            return t.ppf(1-.05/2,df_a)
        
    def __mensagem_erro_11(self):
        return print('Erro11: Parâmetros inválidos.')
    
    def __calcular_erro_efeito(self):
        return 2*self.__erro_exp()/(self.y.shape[0]*2**self.k)**0.5
    
    def erro_efeito(self):
        """Retorna o valor de erro de um efeito"""
        if self.k == None or self.y.all() == None:
            return self.__mensagem_erro_11()
        else:
            return self.__calcular_erro_efeito()
    
    def __calcular_SSPE(self):
     
        return np.sum((self.__array() - np.mean(self.__array()))**2)
    
    def SSPE(self):     
        """Retorna o valor da Soma Quadrática do Erro Puro"""
        if self.y.all() == None:
            return self.__mensagem_erro_11()
        else:
            return self.__calcular_SSPE()
    
    def  df_SSPE(self):
        """Retorna os graus de liberdade da SSPE."""
        return len(self.y)

class Regression2:
    """
   Classe -> Regression2(X, y, SSPE, df) - Cria um modelo de regressão e realiza ajuste do mesmo através de Analisys of Variance
       
   Função adaptada do "fabi_efeito" utilizada no Octave pertencente ao Prof.Dr.Edenir Pereira Filho para o Python**
   Canal do Youtube: https://www.youtube.com/c/EdenirPereiraFilho
       
   Essa rotina tem como finalidade calcular modelos de regressão empregando a seguinte equação:
        
   $inv(X^tX)X^ty$


   Atributes
   -----------
       
   X: matriz com os coeficientes que serao calculados (type: andas.Dataframe)
        
   y: resposta que sera modelada (pandas.Series)
        
   SSPE (optional): Soma Quadrática do Erro Puro dos valores do ponto Central (type: float or int) 
   -> Utilize pde.CP(yc).SSPE() para calcular) --> help(pde.CP.SSPE) para
        
   df (optional): Graus de liberdade do ponto central (type: int)
   -> Utilize pde.CP(yc,k).df_SSPE() --> help(pde.CP.df_SSPE)
   
    ATENÇÃO! ESTE RECURSO ESTÁ AINDA EM DESENVOLVIMENTO E NÃO É FUNCIONAL QUANDO HÁ RÉPLICAS AOS DADOS!
   auto (optional): Automatizar a exclusão dos coeficientes significantes (type: bool)
   -> Sobre mais: help(pde.Regression2.auto).
   
   self_check (optional): Automatizar a verificação se há falta de ajuste do modelo através da análise de variância. 
   -> Sobre mais: help(pde.Regression2.self_check) 
   
   
   Methods
   -----------
        
   create_table_anova: retorna tabela ANOVA do modelo criado (type: NoneType)
   --> help(pde.Regression2.create_table_anova)
    
   plot_graphs_anova: retorna gráficos com os parâmetros da Tabela ANOVA (type: NoneType)
   --> help(pde.Regression2.plot_graphs_anova)
        
   plot_graphs_regression: retorna gráficos do modelo de regressão (type: NoneType)
   --> help(pde.Regression2.plot_graphs_regression)
        
   model_coefients: retorna uma lista com os coeficientes do modelo, quando há coeficientes insignificantes, estes possuirão valor nulo.
   
   recalculate_coefs : retorna um pandas.dataframe com os coeficientes significantes do modelo 
   
   regression2: função mestre que cria um modelo de regressão e realiza ajuste do mesmo através de Analisys of Variance
   --> help(pde.Regression2.regression2) 
   
  
    """
   
    __check_ci = True    
    __final_msg = '\033[1mOperação finalizada! Verifique os resultados em seu diretório.'
    
    def __init__(self, X:object, y:object, SSPE=None, df=None, auto=False, self_check=False):
        self._X = X
        self.y = y
        self.SSPE = SSPE
        self.df = df 
        self._auto = auto
        self._self_check = self_check
            
    def __n_exp(self):
        return  self.X.shape[0]
    
    @property
    def self_check(self):
        return self._self_check
    
    @self_check.setter
    def self_check(self, value:bool) -> bool:
        if isinstance(value,bool):
            self._self_check = value
            
    @property
    def auto(self):
        return self._auto
    
    @auto.setter # changes auto False to True for exclude columns insignificants after regression2 function
    def auto(self, value:bool) -> bool:
        if isinstance(value,bool):
            self._auto = value
    
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, new_dataframe:object) -> object:
        if isinstance(new_dataframe,object):
            self._X = new_dataframe
        
    def __n_coef(self):
        return self.X.shape[1]
    
    def __matrix_X(self):
        return self._X.values 
    
    def __array_y(self):
        return self.y.values
    
    def __calculate_var_coefs(self):
        """
        Retorna valores de variâncias dos coeficientes
        
        Equação aplicada: diag(inv(X'*X))
        """
        return np.diagonal(np.linalg.inv(np.matmul(self.__matrix_X().T,self.__matrix_X()))).round(3)
    
    def __calculate_matrix_coef(self):
        """
        Retorna uma matriz com o resultado da equação abaixo:
        
        b = inv(X'*X))*(X'*Y)
        """
        return np.matmul(np.linalg.inv(np.matmul(self.__matrix_X().T,self.__matrix_X())),
                         self.__matrix_X().T*self.__array_y()).T
    
    def calculate_coefs(self):
        """Retorna a soma dos resultado da definição "__matrix_coef" """
        return np.einsum('ij->j', self.__calculate_matrix_coef()).round(5)
    
    
    def __calculate_pred_values(self):
        """Retorna os valores previstos pelo modelo"""
        return np.matmul(self.X,self.calculate_coefs())
    
 
    def predict(self, value=0):
        """Retorna os valores previstos pelo modelo"""
        return np.matmul(self.X,self.calculate_coefs()+value)
    
    def __calculate_residuals(self):
        """Retorna o valor dos resíduos dos valores previstos"""
        return self.__array_y()-self.__calculate_pred_values()
    
    # Sum of Squares - Part 1
    
    def __calculate_SSreg(self):
        return np.sum((self. __calculate_pred_values()-self.__array_y().mean())**2)
    
    def __calculate_SSres(self):
        return np.sum(self.__calculate_residuals()**2)

    def __calculate_SSTot(self):
        return np.sum(self.__calculate_SSreg()+self.__calculate_SSres())
    
    def __calculate_SSLoF(self):
        return self.__calculate_SSres()-self.SSPE
    
    def __calculate_R2(self):  
        return self.__calculate_SSreg()/self.__calculate_SSTot()
        
    def __calculate_R2_max(self):
        return (self.__calculate_SSTot()-self.SSPE)/self.__calculate_SSTot()
        
    def __calculate_R(self):
        return self.__calculate_R2()**.5
    
    def __calculate_R_max(self):
        return self.__calculate_R2_max()**0.5
    

    # Sum of Squares - Part 2 (deggres of freedom)
    
    def __df_SSreg(self):
        return self.__n_coef()-1
    
    def __df_SSres(self):
        return self.__n_exp()-self.__n_coef()
    
    def __df_SSTot(self):
        return self.__n_exp()-1
    
    def __df_SSLof(self):
        return (self.__n_exp() - self.__n_coef()) - self.df
    
    # Mean of Squares - Part 3
    
    def __calculate_MSreg(self):
        return self.__calculate_SSreg()/self.__df_SSreg()
    
    def __calculate_MSres(self):
        return self.__calculate_SSres()/self.__df_SSres()
    
    def __calculate_MSTot(self):
        return self.__calculate_SSTot()/self.__df_SSTot()
    
    def __calculate_MSPE(self):
        return self.SSPE/self.df
    
    def __calculate_MSLoF(self):
        if self.__df_SSLof() != 0:
            return self.__calculate_SSLoF()/self.__df_SSLof()
        else:
            return self.__calculate_SSLoF()
    
    # F Tests
    
    def __ftest1(self):
        return self.__calculate_MSreg()/self.__calculate_MSres()
    
    def __ftest2(self):
        return self.__calculate_MSLoF()/self.__calculate_MSPE()
    
    # F table
    
    def __ftable1(self): 
        return f.ppf(.95, self.__df_SSreg(),self.__df_SSres()) #F1 tabelado com 95% de confiança
    
    def __ftable2(self): 
        return f.ppf(.95, self.__df_SSLof(),self.df) #F1 tabelado com 95% de confiança
    
    # ANOVA Table
    def __anova_list(self):
        """Formatação da tabela ANOVA"""
        return [
        ['\033[1m'+'Parâmetro','Soma Quadrática (SQ)','Graus de Liberdade(GL)','Média Quadrática (MQ)','Teste F1'+'\033[0m'],
        ['\033[1mRegressão:\033[0m','%.0f'%self.__calculate_SSreg(),self.__df_SSreg(),'%.0f'%self.__calculate_MSreg(),'%.1f'%self.__ftest1() ],
        ['\033[1mResíduo:\033[0m', '%.1f'%self.__calculate_SSres().round(2), self.__df_SSres(),'%.2f'%self.__calculate_MSres(),'%.1f'%self.__ftest1()],
        ['\033[1mTotal:\033[0m', '%.0f'%self.__calculate_SSTot(), self.__df_SSTot(), '%.0f'%self.__calculate_MSTot(), '\033[1mTeste F2\033[0m'],
        ['\033[1mErro puro:\033[0m','%.2f'%self.SSPE, self.df, '%.2f'%self.__calculate_MSPE(), '%.2f'%self.__ftest2() ],
        ['\033[1mFalta de Ajuste:\033[0m', '%.2f'%self.__calculate_SSLoF(), self.__df_SSLof(), '%.2f'%self.__calculate_MSLoF(), '\033[1mF Tabelado\033[0m'],
        ['\033[1mR²:\033[0m', '%.4f'%self.__calculate_R2(), '\033[1mR:\033[0m', '%.4f'%self.__calculate_R(),'F1: %.3f'%self.__ftable1() ],
        ['\033[1mR² máximo:\033[0m','%.4f'%self.__calculate_R2_max(), '\033[1mR máximo:\033[0m', '%.4f'%self.__calculate_R_max(),'F2: %.3f'%self.__ftable2()]
        ]
        
    def create_table_anova(self,show=False):
        """Retorna Nonetype contendo a tabela ANOVA"""
        if show == True:
            return tabulate(self.__anova_list())
        else:
            print('{:^110}'.format('\033[1m'+'TABELA ANOVA'+'\033[0m'))
            print('-='*53)
            print(tabulate(self.__anova_list(),tablefmt="grid"))
            print('-='*53)

    #Data visualization 
    
    def plot_graphs_anova(self):
        """
        Retorna os gráficos referentes aos parâmetros da tabela ANOVA com o objetivo de análise visual.
        
        Returns
        ---------
        1 - Gráfico de Médias Quadráticas: 
        
            - MQ da Regressão
            - MQ dos Resíduos e seu respectivo valor de t-Student
            - MQ do Erro Puro
            - MQ de Falta de Ajuste e seu respectivo valor de t-Student
        
        2 - Gráfico de Teste F2 - MSLof/MSPE:
        
            - Valor de F2 
            - Valor de F tabelado 
            - Relação entre F2/Ftabelado
        
        3 - Gráfico de Teste F1 - MSReg/MSRes:
        
            - Valor de F1 
            - Valor de F tabelado 
            - Relação entre F1/Ftabelado
            
        4 - Gráfico de Coeficiente de Determinação:
            
            - Variação explicada 
            - Variação explicada máxima
        """
        fig = plt.figure(constrained_layout=True,figsize=(10,10))
        subfigs = fig.subfigures(2, 2, wspace=0.07, width_ratios=[1.4, 1.])

        #Mean of Squares (Médias Quadraticas)
        axs0 = subfigs[0,0].subplots(2, 2)

        axs0[0,0].bar('MSReg',self.__calculate_MSreg(),color='darkgreen' ,)
        axs0[0,0].set_title('MQ da Regressão',fontweight='black')
        axs0[0,0].text(-.35, 200, '%.1f'%self.__calculate_MSreg(), fontsize=20,color='white')

        axs0[0,1].bar('MSRes e t',self.__calculate_MSres(),color='darkorange')
        axs0[0,1].set_title('MQ ds Resíduos',fontweight='black')
        axs0[0,1].text(-.35,.5, '%.1f  %.3f'%(self.__calculate_MSres(),CP().invt(self.__df_SSres()-1)), fontsize=20,color='k')
        #axs0[0,1].text(-.35, 2.07, '%.4f'%CP().invt(self.__df_SSres()-1), fontsize=20,color='k')

        axs0[1,0].bar('MSPE',3, color= 'darkred')
        axs0[1,0].set_title('MQ do Erro Puro',fontweight='black')
        axs0[1,0].text(-.35, 1.27,'%.2f'%self.__calculate_MSPE(), fontsize=20,color='w')

        axs0[1,1].bar('MSLoF e t',3,color= 'darkviolet')
        axs0[1,1].set_title('MQ da Falta de Ajuste',fontweight='black')
        axs0[1,1].text(-.35, 1.98, '%.1f'%self.__calculate_MSLoF(), fontsize=20,color='w')
        axs0[1,1].text(-.35, 1.07, '%.4f'%CP().invt(self.__df_SSLof()), fontsize=20,color='w')

        
        #F2 tests (testes F)
        axs1 = subfigs[0,1].subplots(1, 3)

        axs1[0].bar('MSLof/MSPE',self.__ftest2(),color='darkred' ,)
        axs1[0].set_title('Teste F2',fontweight='black')

        axs1[1].bar('F2',self.__ftable2(),color='darkred')
        axs1[1].set_title('F2 tabelado',fontweight='black')

        axs1[2].bar('F2calc/ Ftable',self.__ftest2()/self.__ftable2(), color= 'darkred')
        axs1[2].set_title(r'$\bf\frac{F2_{calculado}}{F2_{tabelado}}$',fontweight='black',fontsize=16,y=1.031)
        axs1[2].axhline(1,color='black')

        #F1 tests (testes F)
        axs2 = subfigs[1,0].subplots(1, 3)

        axs2[0].bar('MSReg/MSRes',self.__ftest1(),color='navy' ,)
        axs2[0].set_title('Teste F1',fontweight='black')

        axs2[1].bar('F1',self.__ftable1(),color='navy')
        axs2[1].set_title('F1 tabelado',fontweight='black')

        axs2[2].bar('F1calc/ Ftable',self.__ftest1()/self.__ftable1(), color= 'navy')
        axs2[2].set_title(r'$\bf\frac{F1_{calculado}}{F1_{tabelado}}$',fontweight='black',fontsize=16,y=1.031)#F1 calculado/\nF1 tabelado
        axs2[2].axhline(1,color='w')
        
        #Coeficiente de determinação 
        axs3 = subfigs[1,1].subplots(1, 2)
        axs3[0].bar('R²',self.__calculate_R2(),color='dimgray' ,)
        axs3[0].set_title('Variação explicada',fontweight='black')
        axs3[0].axhline(1,color='k')
        
        axs3[1].bar('R² max',self.__calculate_R2_max(),color='dimgray')
        axs3[1].set_title('Máxima\n variação explicada',fontweight='black')
        axs3[1].axhline(1,color='k')
        
     
        fig.suptitle('Tabela ANOVA (Analisys of Variance)', fontsize=20, fontweight='black',y=1.05)
        plt.savefig('Tabela ANOVA (Analisys of Variance).png',transparent=True)

      
        return plt.show()

    
    #  Verification of regression coefficients

    
    def  __user_message(self):
        return input('\n\n'+'\033[1mO modelo possui falta de ajuste? [S/N]  \033[0m'+'\n\n')
    
    def __check_model(self): #Return boolean variable for define confidence interval through user message
        check_answer = self.__user_message().upper()
        if check_answer == 'S':
            return True
        elif check_answer == 'N': # this change will be importani in recalculate_model function for decide confidence interval
            return False
        else:
            print('\033[1mErro21: somente as respostas "S" ou "N" serão aceitos.')
            print('Operação Finalizada')
            return sys.exit()
        
    def __self_turning(self, msg=False):
        if (self.__ftest1() > self.__ftable1()) or (self.__ftest2() < self.__ftable2()):
            if msg == True:
                display(Latex(f'$$O\;modelo\;nao\;possui\;falta\;de\;ajuste$$'))
            return False
        else:
            if msg == True:
                display(Latex(f'$$O\;modelo\;possui\;falta\;de\;ajuste$$'))
            return True
    
    
    def define_ic_coefs(self,msg=False): #decides if will calculate ic mslof or msres
        if self.self_check == False:
            check_answer = self.__check_model() #Returns True or False through this method
        elif self.self_check == True:
            check_answer = self.__self_turning(msg)
        else:
            raise TypeError('"pde.Regression2().regression2()" está faltando 1 argumento posicionais requirido "self.check".')
           
        if check_answer == True:
            Regression2.__check_ci = True # this change will be important in recalculate_model function for decide confidence interval
            return self.__define_ic_MSLoF() # to calculate interval confidence for lack of fit
        elif check_answer == False:
            Regression2.__check_ci = False # this change will be important in recalculate_model function for decide confidence interval
            return self.__define_ic_MSRes() #to calculate interval confidence for residues 
        
    def show_ci(self, manual=None):
        """Valores, em módulo, do intervalo de confiança do modelo"""
        if Regression2.__check_ci == True or manual ==True:
            return self.__define_ic_MSLoF()
        elif  Regression2.__check_ci == False or manual == False:
            return self.__define_ic_MSRes()
    
    def __define_ic_MSLoF(self): #calculates confidence interval for mslof
        return (((self.__calculate_MSLoF()*self.__calculate_var_coefs())**0.5)*CP().invt(self.__df_SSLof()-1)).round(4)
        
    def __define_ic_MSRes(self): #calculates confidence interval for msresN
        return (((self.__calculate_MSres()*self.__calculate_var_coefs())**0.5)*CP().invt(self.__df_SSres()-1)).round(4)
    
    def plot_graphs_regression(self):
        """
        Retorna gráficos do modelo de regressão para análise de variáveis insignificantes ao modelo.
        
        Returns
        --------
        
        1 - Gráfico valores Experimental x Previsto e seus respectivos intervalos de confiança.
        
        2 - Gráfico de Previsto x Resíduo
        
        3 - Gráfico de Histograma de resíduos
        
        4 - Gráfico de Coeficientes de Regressão e seus respectivos intervalos de confiança.
        
        
        """
        fig = plt.figure(constrained_layout=True,figsize=(10,14))
        subfigs = fig.subfigures(3,1)
        spec = fig.add_gridspec(3, 2)
        
      
        axs0 =  fig.add_subplot(spec[0, :])
        
        m, b, r_value, p_value, std_err = linregress(self.y, self.__calculate_pred_values())
        
        axs0.plot(self.y, m*self.y + b,color='darkred')
        axs0.legend(['y = {0:.3f}x + {1:.3f}'.format(m,b) +'\n'+'R= {0:.4f}'.format((r_value)**.5)])
        axs0.scatter(self.y,self.predict(),color='b',marker=">",s=40)
        #axs0.scatter(self.y,self.predict(-self.show_ci()),color='b',marker="+",s=20)
        axs0.set_title('Experimental x Previsto',fontweight='black')
        axs0.set_ylabel('Previsto')
        axs0.set_xlabel('Experimental')
        axs0.grid()
        
        axs1 =  fig.add_subplot(spec[1, 0])
        
        axs1.scatter(x=self.__calculate_pred_values(), y=self.__calculate_residuals(),marker="s",color='r')
        axs1.set_title('Previsto x Resíduo',fontweight='black')
        axs1.set_xlabel('Previsto')
        axs1.set_ylabel('Resíduo')
        axs1.axhline(0,color='darkred')
        axs1.grid()
        
        axs2 = fig.add_subplot(spec[1, 1])
        
        axs2.hist(self.__calculate_residuals(),color ='indigo',bins=30)
        axs2.set_title('Histograma dos resduos',fontweight='black')
        axs2.set_ylabel('Frequência')
        axs2.set_xlabel('Resíduos')
        
        #axs3 = fig.add_subplot(spec[2, :])
       
        axs3 =  fig.add_subplot(spec[2, :])
        
        axs3.errorbar(self.X.columns,self.calculate_coefs(),self.define_ic_coefs(True), fmt='^', linewidth=2, capsize=6, color='darkred')
        axs3.axhline(0,color='darkred', linestyle='dashed')
        axs3.set_ylabel('Valores dos coeficientes')
        axs3.set_xlabel('Coeficientes')
        axs3.set_title('Coeficientes de Regressão',fontweight='black')
        axs3.grid()
        
        fig.suptitle('Modelo de Regressão'+'\n' + '-- Regression2 --', fontsize=20, fontweight='black',y=1.1)
        plt.savefig('Modelo de Regressão.png',transparent=True)
        
        return plt.show()
    
    
    # Recalculate the model and to variables insignificant excludes automatically    
    
    def dict_coefs_ci(self): #list with dicts {'coefs':values,'coefs_max':values,'coefs_min':values}
        return  [dict(zip(self.X.columns, self.calculate_coefs())),
                 dict(zip(self.X.columns, self.__calculate_inter_max_min_coefs()[0].round(4))),
                 dict(zip(self.X.columns, self.__calculate_inter_max_min_coefs()[1].round(4)))]
    
    def recalculate_coefs(self):  #returns an array with coefs values and coefs insignificants equal zero 
        """Retorna um DataFrame com os coeficientes significantes"""
        return self.__delete_coefs_insignificants_matrix()
       
    def __calculate_inter_max_min_coefs(self): #returns a tuple with (coef+ci,coef-ci)
        if Regression2.__check_ci == True:
            return [self.calculate_coefs() + self.__define_ic_MSLoF(), 
                            self.calculate_coefs() - self.__define_ic_MSLoF()]
        elif  Regression2.__check_ci == False:
            return [self.calculate_coefs() + self.__define_ic_MSRes(),
                            self.calculate_coefs() - self.__define_ic_MSRes()] 
    
        
    def __delete_coefs_insignificants(self): #select (coef - ci <= coef <= coef + ci) and replace for zero
        coefs = self.dict_coefs_ci()[0]
        max_ = self.dict_coefs_ci()[1]
        min_ = self.dict_coefs_ci()[2]
        for coef in coefs.keys():
            if min_[coef]<= 0 <= max_[coef]:
                coefs[coef] = 0
        return coefs
    
    def model_coefients(self):
        """Retorna uma lista com os valores de coeficientes significantes e valor nulos para coeficientes insignificantes"""
        return list(self.__delete_coefs_insignificants().values())
            
        
    def __delete_coefs_insignificants_matrix(self):
        coefs_recalculate = self.__delete_coefs_insignificants()
        for coef in self.__delete_coefs_insignificants().keys(): # scroll through dictionary keys
            if coefs_recalculate[coef] == 0: # values coef equal zero multiplies the column in matrix X  
                del self.X[coef] #save in local variable for process
        return self.X # changes atribute default
    
        
    def __executor_regression2(self):
        self.create_table_anova()
        self.plot_graphs_anova()
        self.plot_graphs_regression()
    
    def save_dataset(self):
        file = pd.ExcelWriter('dataset.xlsx')
        #coeficiente e intervalo de confiança
        coefs_ci = pd.DataFrame({'Coef': self.model_coefients(),
                                 'coefs-ci':self.model_coefients()-self.define_ic_coefs(),
                                 'coefs+ci':self.model_coefients()+self.define_ic_coefs(),
                                 'C.I': self.define_ic_coefs(),}, index=self.X.columns)
        
        # anova
        anova = pd.DataFrame([
        [' Regressão: ',self.__calculate_SSreg(),self.__df_SSreg(),self.__calculate_MSreg(),self.__ftest1() ],
        [' Resíduo: ',self.__calculate_SSres(), self.__df_SSres(),self.__calculate_MSres(),self.__ftest1()],
        [' Total: ', self.__calculate_SSTot(), self.__df_SSTot(),self.__calculate_MSTot(), ' Teste F2 '],
        [' Erro puro: ',self.SSPE, self.df, self.__calculate_MSPE(),self.__ftest2() ],
        [' Falta de Ajuste: ', self.__calculate_SSLoF(), self.__df_SSLof(), self.__calculate_MSLoF(), ' F Tabelado '],
        [' R²: ', self.__calculate_R2(), ' R: ', self.__calculate_R(), self.__ftable1() ],
        [' R² máximo: ',self.__calculate_R2_max(), ' R máximo: ', self.__calculate_R_max(),self.__ftable2()]
        ], columns=['Parâmetro','Soma Quadrática (SQ)','Graus de Liberdade(GL)','Média Quadrática (MQ)','Teste F1'])
        
        # previsto x experimental x coefs codficados
        pred_exp = pd.DataFrame(self.__matrix_X(), columns=self.X.columns)
        pred_exp['Experimental'] = self.y
        pred_exp['Previsto'] = self.__calculate_pred_values()
        
        #savando dados em abas dentro do arquivo
        #anova.to_excel('ANOVA.xlsx')
        #coefs_ci.to_excel('coefs_ic.xlsx')
        #pred_exp.to_excel('exp_pred.xlsx')
        anova.to_excel(file,sheet_name= 'ANOVA')
        coefs_ci.to_excel(file,sheet_name='coefs_ic')
        pred_exp.to_excel(file,sheet_name='exp_pred')
        return file.save()
    
    
    def regression2(self):
        
        """
        Função -> regression2
        
        Função adaptada do "fabi_efeito" utilizada no Octave pertencente ao Prof.Dr.Edenir Pereira Filho para o Python**
        Canal do Youtube: https://www.youtube.com/c/EdenirPereiraFilho
        
        Essa rotina tem como finalidade calcular modelos de regressão empregando a seguinte equação:
        
        inv(X^tX)X^ty


        Atributes - Inseridos na instancia da classe Regression2
        -----------
        
        X = matriz com os coeficientes que serao calculados (type: andas.Dataframe)
        
        y = resposta que sera modelada (pandas.Series)
        
        SSPE = Soma Quadrática do Erro Puro dos valores do ponto Central (type: float or int) 
            -> Utilize pde.CP(yc).SSPE() para calcular) --> help(pde.CP.SSPE) para
        
        df = Graus de liberdade do ponto central (type: int)
            -> Utilize pde.CP(yc,k).df_SSPE() --> help(pde.CP.df_SSPE)
          
        ATENÇÃO! ESTE RECURSO ESTÁ AINDA EM DESENVOLVIMENTO E NÃO É FUNCIONAL QUANDO HÁ RÉPLICAS AOS DADOS!
       auto (optional): Automatizar a exclusão dos coeficientes significantes (type: bool)
       -> Sobre mais: help(pde.Regression2.auto).
   
       self_check (optional): Automatizar a verificação se há falta de ajuste do modelo através da análise de variância. 
       -> Sobre mais: help(pde.Regression2.self_check) 
        
        Returns
        -----------
        
        1 - Tabela ANOVA (Analisys of Variance) (type: NoneType)
        
        2- plot_graphs_anova() (type: NoneType) --> help(pde.Regression2.plot_graphs_anova)
        
        3 - Interação com usuário perguntando se há falta de ajuste no modelo. (type: str)
        
        4- plot_graphs_regression() (type: NoneType) --> help(pde.Regression2.plot_graphs_regression)
        
        
        """
        
        self.__executor_regression2()
        if self.auto == True:
            self.recalculate_coefs()
            self.__executor_regression2()
            return print(Regression2.__final_msg)
        else:
            return print(Regression2.__final_msg)
        
        
class Super_fabi:
    """
    Funcao para calcular superficie de resposta e gráfico de contorno
    A matriz X deve conter:
    Coluna 1 = coeficientes na seguinte ordem, b0, b1, b2, b11, b22, b12
    Coluna 2 = Valores codificados da variavel 1
    Coluna 3 = Valores reais da variável 1
    Coluna 4 = Valores codificados da variavel 2
    Coluna 5 = Valores reais da variavel 2
    """
    def __init__(self, coefs:list,realmax1=None,realmin1=None,realmax2=None,realmin2=None,
                 codmax1=None,codmin1=None,codmax2=None,codmin2=None):
        self.coefs = coefs
        self.realmax1 = realmax1
        self.realmin1 = realmin1
        self.realmax2 = realmax2
        self.realmin2 = realmin2
        self.codmax1 = codmax1 
        self.codmin1 = codmin1 
        self.codmax2 = codmax2 
        self.codmin2 = codmin2 
    

    def array_n1(self):
        return np.linspace(self.codmin1 ,self.codmax1 ,num=101)
    
    def array_n2(self):
        return np.linspace(self.codmin2,self.codmax2,num=101)
    
    def array_r1(self):
        return np.linspace(self.realmin1 ,self.realmax1 ,num=101)
    
    def array_r2(self):
        return np.linspace(self.realmin2,self.realmax2,num=101)
    
    def meshgrid_cod(self):
        return np.meshgrid(self.array_n1(),self.array_n2())
    
    def meshgrid_real(self):
        return np.meshgrid(self.array_r1(),self.array_r2())
    
    def z(self, meshgrid=None, x=None, y=None, manual=False):
    
        """
        Retorna valor previsto pelo modelo.
        
        Parameters 
        -----------
        
        v1: valor(es) da variável 1 (if meshgrid is True --> type numpy.array else: type float)
        
        v2: valor(es) da variável 2 (if meshgrid is True --> type numpy.array else: type float)
        
        n_var: número de variáveis a serem analisadas, por padrão n_var=2 
        
        meshgrid: (optional) (if meshgrid == True --> will be  created a matrix with Z values else: returns only a item type float)

        manual: (optional) (if manual is True --> will be calculate z value for x and y parameters ) (type:bool)
        
        """
        b0, b1, b2, b11, b22, b12 = self.coefs
        
        
        if manual == True: # if manual mode to be activate
            if x == None or y == None: # check parameters for manual mode
                raise TypeError('recalculate_coefs() está faltando 2 argumentos posicionais requirido "x" e "y".')
            else:       
                    return (b0 + b1*x + b2*y + b11*x**2 + b22*y**2 + b12*x*y).round(4)
                    
        elif meshgrid == None and manual == False:
            raise TypeError('Insira parâmetros ao método.')
            
        try:
            if meshgrid == True:
                x, y = self.meshgrid_cod()
                return (b0 + b1*x + b2*y + b11*x**2 + b22*y**2 + b12*x*y).round(4)
            elif meshgrid == False:
                x, y = self.array_n1(), self.array_n2()
                return (b0 + b1*x + b2*y + b11*x**2 + b22*y**2 + b12*x*y).round(4)
                    
        except: raise TypeError('recalculate_coefs() está faltando 1 argumento posicional requirido "meshgrid".')

    def __index_max_values(self): # Return matrix meshgrid index for max value model
        idx1, idx2 = np.where(self.z(meshgrid=True) == self.z(meshgrid=True).max().max())
        return idx1[0],idx2[0]  
    
    @property
    def maxcod(self):
        """Retorna valores das coordenadas do sinal máximo para as variáveis codificadas.
        
        Returns 
        ----------
        
        (x_coordenate, y_coordenate) for codificates values  like a tuple.
        
        """
        idx1, idx2 = self.__index_max_values()[0], self.__index_max_values()[1]
        v2, v1 =  self.meshgrid_cod()[0], self.meshgrid_cod()[1]
        return v1[idx2][idx1], v2[idx2][idx1]
    
    @property
    def maxreal(self):
        """Retorna valores das coordenadas do sinal máximo para as variáveis codificadas.
        
        Returns 
        ----------
        
        (x_coordenate, y_coordenate) for real values like a tuple.
        """
        idx1, idx2 = self.__index_max_values()[0], self.__index_max_values()[1]
        v1, v2 = self.meshgrid_real()[0], self.meshgrid_real()[1]
        return v1[idx1][idx2], v2[idx1][idx2]   
 
    @property
    def zmax(self):
        r"""Retorna o valor do sinal máximo do modelo.
        
        Return
        --------
        
        fmax(x,y) = zmax like a float.
        """
        return self.z(meshgrid=True).max().max()
    
    
 
    def __etiqueta(self,matrix_X, vector_y, ax):
        vector_y = [str(j) for j in vector_y.values] # vector_y to string list 
               
        for i, label in enumerate(vector_y):
            ax.annotate(label,( matrix_X['b1'][i], matrix_X['b2'][i]),color='k',fontsize=10)  
    
    def superficie(self, matrix_X = None, vector_y = None,scatter=False):
        """Retorna os gráficos de superfície e de contorno do modelo
        
        Parameters
        ------------
        
        X: matriz X com os valores codificados dos coeficiente (type: pandas.dataframe)
        
        y: vetor y com os valores de sinais 
        
        """
        fig = plt.figure(figsize=(12,12))
        
        # Superficie de resposta
        ax1 = fig.add_subplot(1,2, 1, projection='3d')

        
        V1, V2= self.meshgrid_real()
        Z = self.z(meshgrid=True)
        b0, b1, b2, b11, b22, b12 = self.coefs
        
        surf =  ax1.plot_surface(V1, V2, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
        
        ax1.set_title('Superfície de Resposta do Modelo',fontsize=12, fontweight='black',y=1,x=.55)
        ax1.set_xlabel('Variável 1')
        ax1.set_ylabel('Variável 2')
        ax1.set_zlabel('Resposta do Modelo')
        
        # Contorno 
        v1, v2= self.meshgrid_cod()
        
        ax2 = fig.add_subplot(1,2,2)
        
        contours = ax2.contour(v1, v2, Z, 3,colors='black', levels=6)
        ax2.clabel(contours, inline=True, fontsize=12)
        plt.imshow(Z, extent=[self.codmin1, self.codmax1, self.codmax2, self.codmin2],  cmap='viridis', alpha=1)
        plt.colorbar(aspect=6, pad=.15)
        
        
        if scatter == True:
            if isinstance(matrix_X,object):
                ax2.scatter(matrix_X['b1'],matrix_X['b2'], color='w',marker=(5, 1),s=50)
                self.__etiqueta(matrix_X, vector_y, ax2)
        
        ax2.scatter(self.maxcod[0], self.maxcod[1], color='darkred',marker=(5, 1),s=100)   
        ax2.annotate(r'$z_{max}= %.2f$'%self.zmax, (self.maxcod[0], self.maxcod[1]),color='k')
        
        ax2.set_xticks(list(np.arange(self.codmin1,round((2*self.codmax1),4),step=self.codmax1)) + [round(self.maxcod[0],3)])
        ax2.set_yticks(list(np.arange(self.codmin2,round((2*self.codmax2),4),step=self.codmax2)) + [round(self.maxcod[1],3)])
        ax2.set_xlabel('Variável 1')
        ax2.set_ylabel('Variável 2')
        ax2.set_title('Contorno do Modelo',fontsize=12, fontweight='black',y=1.1)
        
        fig.text(0.2,.71,r'$R_{max}(%.2f,%.2f) = %.1f\qquad\qquad\qquad  v_1^{max} = %.1f \quad e\quad v_2^{max} = %.1f $'%(self.maxcod[0],self.maxcod[1],self.zmax,self.maxreal[0],self.maxreal[1]),
                 fontsize=15,horizontalalignment='left')
        plt.suptitle(r'$\bfResposta = {} + {}v_1 + {}v_2 + {}v_1^2 + {}v_2^2 + {}v_1v_2 $'.format(b0, b1, b2, b11, b22, b12),y=.77,x=.45,fontsize=20)
        
        plt.tight_layout(w_pad=5)
        plt.show()
        

    def solver_diff(self, k=2, printf=False):
        """Método que retorna o valor máximo de resposta e os respectivos valores codificados das variáveis exploratórias do modelo através dos cálculos das derivada parciais de primeira ordem. 
        Selecione o número de variáveis através do parâmetro k. 
        Esta função é capaz de calcular para modelos com 2,3 ou 4 variáveis. 

        Parameters
        ------------

        k: número de variáveis do modelo (type: int) 

        printf (optional): Por padrão (False), será retornado valores de coordendas e resposta máxima em uma tupla e quando printf=True será retornado uma mensagem com as resposta em um display em linguagem Latex. 

        Returns
        ------------
        Retorna valores das coordenadas exploratórias para o máximo global do modelo através da derivada parcial.
        """
        v1,v2,v3,v4 = symbols('v1 v2 v3 v4', real=True) 
        
        try:
            if k == 2:           
                b0, b1, b2, b11, b22, b12 = self.coefs
                f = b0 + b1*v1 + b2*v2 + b11*v1**2 + b22*v2**2 + b12*v1*v2
                X = np.matrix([
                     [2*b11,b12],
                     [b12,2*b22]])
                y = -np.array(self.coefs[1:3])

                p_diff = np.array(np.matmul(np.linalg.inv(X),y))[0] #inv(X)*y multiplicação de matriz e vetor (v1,v2,v3,v4)
                fmax = f.subs([(v1,p_diff[0]),(v2,p_diff[1])])

                resultados = np.append(p_diff,fmax)
                
                if printf == False:
                    return pd.DataFrame({'Resultados':resultados},index=['b1','b2','Resposta'])
                else:
                    return display(Latex("$$f'({0:.3f},{1:.3f})= {2:.2f}$$".format(p_diff[0],p_diff[1],fmax)))

            elif k == 3:   
                b0, b1, b2, b3, b11, b22, b33, b12, b13, b23 = self.coefs
                f = b0 + b1*v1 + b2*v2 + b3*v3 + b11*v1**2 + b22*v2**2 + b33*v3**2 + b12*v1*v2 + b13*v1*v3 + b23*v2*v3
                X = np.matrix([
                     [2*b11,b12,b13],
                     [b12,2*b22,b23],
                     [b13,b23,2*b33]])
                y = -np.array(self.coefs[1:4])

                p_diff = np.array(np.matmul(np.linalg.inv(X),y))[0] #inv(X)*y multiplicação de matriz e vetor (v1,v2,v3,v4)
                fmax = f.subs([(v1,p_diff[0]),(v2,p_diff[1]),(v3,p_diff[2])])

                resultados = np.append(p_diff,fmax)

                if printf == False:
                    return pd.DataFrame({'Resultados':resultados},index=['b1','b2','b3','Resposta'])
                else:
                    return display(Latex("$$f'({0:.3f},{1:.3f},{2:.3f})= {3:.2f}$$".format(p_diff[0],p_diff[1],p_diff[2],fmax)))

            elif k == 4:  
                b0, b1, b2, b3, b4, b11, b22, b33, b44, b12, b13, b14, b23, b24, b34 = self.coefs
                f=b0+b1*v1+b2*v2+b3*v3+b4*v4+b11*v1**2+b22*v2**2+b33*v3**2+b44*v4**2+b12*v1*v2+b13*v1*v3+b14*v1*v4+b23*v2*v3+b24*v2*v4+b34*v3*v4
                X = np.matrix([
                     [2*b11,b12,b13,b14],
                     [b12,2*b22,b23,b24],
                     [b13,b23,2*b33,b34],
                     [b14,b24,b34,2*b44]])
                y = -np.array(self.coefs[1:5])

                p_diff = np.array(np.matmul(np.linalg.inv(X),y))[0] #inv(X)*y multiplicação de matriz e vetor (v1,v2,v3,v4)
                fmax = f.subs([(v1,p_diff[0]),(v2,p_diff[1]),(v3,p_diff[2]),(v4,p_diff[3])])

                resultados = np.append(p_diff,fmax)

                if printf == False:
                    return pd.DataFrame( {'Resultados':resultados},index=['b1','b2','b3','b4','Resposta'])
                else:
                    return display(Latex("$$f'({0:.4f},{1:.3f},{2:.3f},{3:.3f})= {4:.2f}$$".format(p_diff[0],p_diff[1],p_diff[2],
                                                                                                   p_diff[3],fmax)))
        except: raise TypeError(f'Não há registro para solução da equação para "k == {k}"')