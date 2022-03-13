#class Fabi_efeito
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
import sys
#class Super_fabi

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
        if self.k == None or self.y == None:
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
       
   X = matriz com os coeficientes que serao calculados (type: andas.Dataframe)
        
   y = resposta que sera modelada (pandas.Series)
        
   SSPE = Soma Quadrática do Erro Puro dos valores do ponto Central (type: float or int) 
   -> Utilize pde.CP(yc).SSPE() para calcular) --> help(pde.CP.SSPE) para
        
   df = Graus de liberdade do ponto central (type: int)
   -> Utilize pde.CP(yc,k).df_SSPE() --> help(pde.CP.df_SSPE)
        
   Methods
   -----------
        
   create_table_anova: retorna tabela ANOVA do modelo criado (type: NoneType)
   --> help(pde.Regression2.create_table_anova)
    
   plot_graphs_anova: retorna gráficos com os parâmetros da Tabela ANOVA (type: NoneType)
   --> help(pde.Regression2.plot_graphs_anova)
        
   plot_graphs_regression: retorna gráficos do modelo de regressão (type: NoneType)
   --> help(pde.Regression2.plot_graphs_regression)
        
   regression2: função mestre que cria um modelo de regressão e realiza ajuste do mesmo através de Analisys of Variance
   --> help(pde.Regression2.regression2) 
    """
        
        
    def __init__(self, X, y, SSPE=None, df=None):
        self.X = X 
        self.y = y
        self.SSPE = SSPE
        self.df = df 
    
    def __n_exp(self):
        return  X.shape[0]
    
    def __n_coef(self):
        return X.shape[1]
    
    def __matrix_X(self):
        return self.X.values 
    
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
    
    def __calculate_coefs(self):
        """Retorna a soma dos resultado da definição "__matrix_coef" """
        return np.einsum('ij->j', self.__calculate_matrix_coef()).round(2)
    
    def __calculate_pred_values(self):
        """Retorna os valores previstos pelo modelo"""
        return np.matmul(self.X,self.__calculate_coefs())
    
    def __calculate_residuals(self):
        """Retorna o valor dos resíduos dos valores previstos"""
        return self.__array_y()-self.__calculate_pred_values()
    
    # Sum of Squares - Part 1
    
    def __calculate_SSreg(self):
        return np.sum((self. __calculate_pred_values()-self.__array_y().mean())**2).round(2)
    
    def __calculate_SSres(self):
        return np.sum(self.__calculate_residuals()**2).round(2)

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
        return (self.__n_exp()-self.__n_coef())-self.df
    
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
        return self.__calculate_SSLoF()/self.__df_SSLof()
    
    # F Tests
    
    def __ftest1(self):
        return self.__calculate_MSreg()/self.__calculate_MSres()
    
    def __ftest2(self):
        return self.__calculate_MSLoF()/self.__calculate_MSPE()
    
    # F table
    
    def __ftable(self): 
        return f.ppf(.95, self.__df_SSreg(),self.__df_SSres()) #F tabelado com 95% de confiança
    
    # ANOVA Table
    def __anova_list(self):
        """Formatação da tabela ANOVA"""
        return [
        ['\033[1m'+'Parâmetro','Soma Quadrática (SQ)','Graus de Liberdade(GL)','Média Quadrática (MQ)','Teste F1'+'\033[0m'],
        ['\033[1mRegressão:\033[0m','%.0f'%self.__calculate_SSreg(),self.__df_SSreg(),'%.0f'%self.__calculate_MSreg(),'%.1f'%self.__ftest1() ],
        ['\033[1mResíduo:\033[0m', '%.1f'%self.__calculate_SSres(), self.__df_SSres(),'%.2f'%self.__calculate_MSres(),'%.1f'%self.__ftest1()],
        ['\033[1mTotal:\033[0m', '%.0f'%self.__calculate_SSTot(), self.__df_SSTot(), '%.0f'%self.__calculate_MSTot(), '\033[1mTeste F2\033[0m'],
        ['\033[1mErro puro:\033[0m','%.2f'%self.SSPE, self.df, '%.2f'%self.__calculate_MSPE(), '%.2f'%self.__ftest2() ],
        ['\033[1mFalta de Ajuste:\033[0m', '%.2f'%self.__calculate_SSLoF(), self.__df_SSLof(), '%.2f'%self.__calculate_MSLoF(), '%.2f'%self.__ftest2()],
        ['\033[1mR²:\033[0m', '%.4f'%self.__calculate_R2(), '\033[1mR:\033[0m', '%.4f'%self.__calculate_R(),  '\033[1mF Tabelado\033[0m'],
        ['\033[1mR² máximo:\033[0m','%.4f'%self.__calculate_R2_max(), '\033[1mR máximo:\033[0m', '%.4f'%self.__calculate_R_max(),'%.0f'%CP(y=self.y).SSPE()]
        ]
        
    def create_table_anova(self):
        """Retorna Nonetype contendo a tabela ANOVA"""
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
        axs0[0,0].text(-.35, 100, '%.1f'%self.__calculate_MSreg(), fontsize=20,color='white')

        axs0[0,1].bar('MSRes e t',self.__calculate_MSres(),color='darkorange')
        axs0[0,1].set_title('MQ ds Resíduos',fontweight='black')
        axs0[0,1].text(-.35, 2.5, '%.1f'%self.__calculate_SSres(), fontsize=20,color='k')
        axs0[0,1].text(-.35, 1.07, '%.4f'%CP().invt(self.__df_SSres()), fontsize=20,color='k')

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

        axs1[1].bar('F2',self.__ftest2(),color='darkred')
        axs1[1].set_title('F tabelado',fontweight='black')

        axs1[2].bar('F2calc/ Ftable',self.__ftest2()/self.__ftable(), color= 'darkred')
        axs1[2].set_title(r'$\bf\frac{F2_{calculado}}{F_{tabelado}}$',fontweight='black',fontsize=16,y=1.031)
        axs1[2].axhline(1,color='black')

        #F1 tests (testes F)
        axs2 = subfigs[1,0].subplots(1, 3)

        axs2[0].bar('MSReg/MSRes',2,color='navy' ,)
        axs2[0].set_title('Teste F1',fontweight='black')

        axs2[1].bar('F1',3,color='navy')
        axs2[1].set_title('F1 tabelado',fontweight='black')

        axs2[2].bar('F1calc/ Ftable',self.__ftest1()/self.__ftable(), color= 'navy')
        axs2[2].set_title(r'$\bf\frac{F1_{calculado}}{F_{tabelado}}$',fontweight='black',fontsize=16,y=1.031)#F1 calculado/\nF1 tabelado
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

    
    # Verificação dos coeficientes de regressão  
    
    @staticmethod 
    def __user_message():
        return input('\n\n'+'\033[1mO modelo possui falta de ajuste? [S/N]  \033[0m'+'\n\n')
    
    def __check_model(self):
        check_answer = self.__user_message().upper()
        if check_answer == 'S':
            return True
        elif check_answer == 'N':
            return False
        else:
            print('\033[1mErro21: somente as respostas "S" ou "N" serão aceitos.')
            print('Operação Finalizada')
            return sys.exit()
    
    def __define_ic_coefs(self):
        check_answer = self.__check_model()
        if check_answer == True:
            return self.__define_ic_MSLoF()
        elif check_answer == False:
            return self.__define_ic_MSRes()
        
    def __define_ic_MSLoF(self):
        return (((self.__calculate_MSLoF()*self.__calculate_var_coefs())**0.5)*CP().invt(self.__df_SSLof()-1)).round(2)
        
    def __define_ic_MSRes(self):
        return (((self.__calculate_MSres()*self.__calculate_var_coefs())**0.5)*CP().invt(self.__df_SSres()-1)).round(2)
    
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
        fig = plt.figure(constrained_layout=True,figsize=(10,10))
        subfigs = fig.subfigures(3,1)
        spec = fig.add_gridspec(3, 2)
        
      
        axs0 =  fig.add_subplot(spec[0, :])
        
        m, b, r_value, p_value, std_err = linregress(self.y, self.__calculate_pred_values())
        axs0.plot(self.y, m*self.y + b,color='darkred')
        axs0.legend(['y = {0:.3f}x + {1:.3f}'.format(m,b) +'\n'+'R²= {0:.4f}'.format(r_value)])
        
        axs0.errorbar(self.y,self.__calculate_pred_values(),self.__calculate_residuals(), fmt='o', linewidth=2, capsize=6, color='darkred')

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
        
        axs3.errorbar(self.X.columns,self.__calculate_coefs(),self.__define_ic_coefs()[0], fmt='o', linewidth=2, capsize=6, color='darkred')
        axs3.axhline(0,color='darkred', linestyle='dashed')
        axs3.set_ylabel('Valores dos coeficientes')
        axs3.set_xlabel('Coeficientes')
        axs3.set_title('Coeficientes de Regressão',fontweight='black')
        axs3.grid()
        
        fig.suptitle('Modelo de Regressão'+'\n' + '-- Regression2 --', fontsize=20, fontweight='black',y=1.1)
        plt.savefig('Modelo de Regressão.png',transparent=True)
        
        return plt.show()
    
    def regression2(self):
        """
        Função -> regression2
        
        Função adaptada do "fabi_efeito" utilizada no Octave pertencente ao Prof.Dr.Edenir Pereira Filho para o Python**
        Canal do Youtube: https://www.youtube.com/c/EdenirPereiraFilho
        
        Essa rotina tem como finalidade calcular modelos de regressão empregando a seguinte equação:
        
        $inv(X^tX)X^ty$


        Atributes - Inseridos na instancia da classe Regression2
        -----------
        
        X = matriz com os coeficientes que serao calculados (type: andas.Dataframe)
        
        y = resposta que sera modelada (pandas.Series)
        
        SSPE = Soma Quadrática do Erro Puro dos valores do ponto Central (type: float or int) 
            -> Utilize pde.CP(yc).SSPE() para calcular) --> help(pde.CP.SSPE) para
        
        df = Graus de liberdade do ponto central (type: int)
            -> Utilize pde.CP(yc,k).df_SSPE() --> help(pde.CP.df_SSPE)
        
        Returns
        -----------
        
        1 - Tabela ANOVA (Analisys of Variance) (type: NoneType)
        
        2- plot_graphs_anova() (type: NoneType) --> help(pde.Regression2.plot_graphs_anova)
        
        3 - Interação com usuário perguntando se há falta de ajuste no modelo. (type: str)
        
        4- plot_graphs_regression() (type: NoneType) --> help(pde.Regression2.plot_graphs_regression)
        
        
        """
        self.create_table_anova()
        self.plot_graphs_anova()
        self.plot_graphs_regression()
        
