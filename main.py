from tkinter import *
from tkinter import filedialog, messagebox, ttk
import customtkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.pipeline import *
from xgboost import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.compose import *
from sklearn.model_selection import *
from sklearn.neighbors import *
from sklearn.compose import make_column_transformer
from category_encoders import *
from sklearn.model_selection import *
import statistics
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tema do customtkinter

customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        self.title("Mycelium")
        self.state('zoomed')

        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Hackathon TEROS",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)

        #self.button_1 = customtkinter.CTkButton(master=self.frame_left,
        #                                        text="Estimativa de Aprovação/Rejeição",
        #                                        fg_color=("gray75", "gray30"),  # <- custom tuple-color
        #                                        command=lambda: upload_arquivo1())
        #                                        #command=self.button_event)
        #self.button_1.grid(row=3, column=0, pady=10, padx=20)

        #self.button_2 = customtkinter.CTkButton(master=self.frame_left,
        #                                        text="Exportar",
        #                                        fg_color=("gray75", "gray30"),  # <- custom tuple-color
        #                                        command=lambda: save_file())
        #self.button_2.grid(row=4, column=0, pady=10, padx=20)

        self.button_3 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Histórico Negocial",
                                                fg_color=("gray75", "gray30"),  # <- custom tuple-color
                                                command=lambda: upload_arquivo2())
        self.button_3.grid(row=2, column=0, pady=10, padx=20)


        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_right.rowconfigure(7, weight=10)
        self.frame_right.columnconfigure((0, 1), weight=1)
        self.frame_right.columnconfigure(2, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.columnconfigure(0, weight=1)

        self.frame_info.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)


        # ============ frame_right ============
        planilha = LabelFrame(self, text="Dados CSV")
        planilha.place(height=400, width=1000, rely=0.05, relx=0.3)

        # Planilha
        resultado = ttk.Treeview(planilha)
        resultado.place(relheight=1,relwidth=1)

        resultadoScrolly = Scrollbar(planilha, orient="vertical", command=resultado.yview)
        resultadoScrollx = Scrollbar(planilha, orient="horizontal", command=resultado.xview)
        resultado.configure(xscrollcommand=resultadoScrollx.set, yscrollcommand=resultadoScrolly.set)
        resultadoScrollx.pack(side="bottom", fill="x")
        resultadoScrolly.pack(side="right", fill="y")

        def save_file():
            file = filedialog.asksaveasfilename(
                filetypes=[("csv", "*.csv")],
                defaultextension=".csv")
            fob = open(file, 'w', newline='')
            csvwriter = csv.writer(fob, delimiter=',')
            for row_id in resultado.get_children():
                row = resultado.item(row_id)['values']
                csvwriter.writerow(row)
            fob.close()
            messagebox.showinfo(title=None, message="Exportação realizada com sucesso.")
        
        def upload_arquivo2():
            f_types = [('CSV files', "*.csv"), ('All', "*.*")]
            arquivo = filedialog.askopenfilename(filetypes=f_types)

            if arquivo:
                try:
                    df_treino = pd.read_csv(arquivo, encoding="ISO-8859-1")
                except ValueError:
                    messagebox.showerror("Informação", "Arquivo selecionado invalido")
                    return None
                except FileNotFoundError:
                    messagebox.showerror("Informação", f"Nenhum arquivo como {arquivo}")
                    return None

            # implementação do modelo 

            # tratamento

            df_treino.drop(columns=['Unnamed: 0', 'Codigo_da_oportunidade', 'ID_cliente', 'Data_de_criacao', 'Data_Real_de_Encerramento_da_Opp2'], inplace=True)
            df_treino['Valor_corrigido2'] = df_treino['Valor_corrigido2'].round(2)
            df_treino = df_treino.replace(np.nan, 0)
            df_treino.Valor_corrigido2 = df_treino.Valor_corrigido2.astype(int)
            df_treino.Custo_Total = df_treino.Custo_Total.astype(int)

            df_dummie_train = pd.get_dummies(df_treino)

            X = df_dummie_train.drop(['id_fechou'], axis=1)
            y = df_dummie_train['id_fechou']

            # treino e teste

            X_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)

            # rf

            rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)  
            rf.fit (X_train, y_train); # Treine o modelo nos dados de treinamento
            preditos_rf = rf.predict(x_test)

            # modelo xgboost
            
            xgb = XGBClassifier(learning_rate =0.1,
                                n_estimators=1000,
                                max_depth=6,
                                min_child_weight=1,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                nthread=4,
                                scale_pos_weight=1.0,
                                seed=27)
            
            xgb.fit(X_train, y_train)
            preditos_xgb = xgb.predict(x_test)

            self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="A Estimar Aprovação/Rejeição",
                                                fg_color=("gray75", "gray30"),  # <- custom tuple-color
                                                command=lambda: upload_arquivo1())
                                                #command=self.button_event)
            self.button_1.grid(row=3, column=0, pady=10, padx=20)


            def upload_arquivo1():
                f_types = [('CSV files', "*.csv"), ('All', "*.*")]
                arquivo = filedialog.askopenfilename(filetypes=f_types)

                if arquivo:
                    try:
                        df_teste = pd.read_csv(arquivo, encoding="ISO-8859-1")
                    except ValueError:
                        messagebox.showerror("Informação", "Arquivo selecionado invalido")
                        return None
                    except FileNotFoundError:
                        messagebox.showerror("Informação", f"Nenhum arquivo como {arquivo}")
                        return None

                df_teste.rename(columns={'ComissÃ£o sobre Parceiros': 'Comissão sobre Parceiros',
                                         'EquilÃ­brio fiscal': 'Equilíbrio fiscal',
                                         'GestÃ£o EstratÃ©gica': 'Gestão Estratégica',
                                         'GestÃ£o da OperaÃ§Ã£o': 'Gestão da Operação',
                                         'GestÃ£o da Receita': 'Gestão da Receita',
                                         'GestÃ£o da SaÃºde': 'Gestão da Saúde',
                                         'GestÃ£o da SeguranÃ§a PÃºblica': 'Gestão da Segurança Pública',
                                         'GestÃ£o da SeguranÃ§a ViÃ¡ria': 'Gestão da Segurança Viária',
                                         'GestÃ£o de Gastos': 'Gestão de Gastos',
                                         'GestÃ£o de operaÃ§Ãµes projetizadas': 'Gestão de operações projetizadas',
                                         'GestÃ£o de pessoas': 'Gestão de pessoas',
                                         'TransformaÃ§Ã£o Digital': 'Transformação Digital'}, inplace = True)

                df_teste_dummie = pd.get_dummies(df_teste)
                
                # voting class
                
                voto = VotingClassifier(estimators=[('xgb',xgb)], voting='soft')
                voto = voto.fit(X_train, y_train)

                y_predito = voto.predict(df_teste_dummie)

                df_teste['id_fechou'] = voto.predict_proba(X_train[X_train.columns])[:,1]
                df_teste[['Codigo_da_oportunidade','id_fechou']]

                # exibir resultado na tela

                resultado["column"] = list(df_teste.columns)
                resultado["show"] = "headings"
                for column in resultado["column"]:
                    resultado.heading(column, text=column)


                figura1 = plt.Figure(figsize=(8, 6), dpi=60)
                grafico1 = figura1.add_subplot(111)
                grafico1.scatter(df_teste['intervalo_tempo'], df_teste['Custo_Total'], color='red')
                plt.title('Custo_Total vs intervalo_tempo', fontsize=14)
                plt.xlabel('intervalo_tempo', fontsize=14)
                plt.ylabel('Custo_Total', fontsize=14)
                canva1 = FigureCanvasTkAgg(figura1, self)
                canva1.get_tk_widget().place(rely=0.5, relx=0.15)

                nomeGraf1 = Label(self, text="intervalo_tempo vs Custo_Total")
                nomeGraf1.place(rely=0.5, relx=0.15)

                figura2 = plt.Figure(figsize=(8, 6), dpi=60)
                grafico2 = figura2.add_subplot(111)
                grafico2.scatter(df_teste['intervalo_tempo'], df_teste['Valor_corrigido2'], color='red')
                plt.title('Valor_corrigido2 vs intervalo_tempo', fontsize=14)
                plt.xlabel('intervalo_tempo', fontsize=14)
                plt.ylabel('Valor_corrigido2', fontsize=14)
                canva2 = FigureCanvasTkAgg(figura2, self)
                canva2.get_tk_widget().place(rely=0.5, relx=0.425)

                nomeGraf2 = Label(self, text="intervalo_tempo vs Valor_corrigido2")
                nomeGraf2.place(rely=0.5, relx=0.425)

                figura3 = plt.Figure(figsize=(8, 6), dpi=60)
                grafico3 = figura3.add_subplot(111)
                grafico3.scatter(df_teste['intervalo_tempo'], df_teste['Margem_Total'], color='red')
                plt.title('Margem_Total vs intervalo_tempo', fontsize=14)
                plt.xlabel('intervalo_tempo', fontsize=14)
                plt.ylabel('Margem_Total', fontsize=14)
                canva3 = FigureCanvasTkAgg(figura3, self)
                canva3.get_tk_widget().place(rely=0.5, relx=0.7)

                nomeGraf3 = Label(self, text="intervalo_tempo vs Margem_Total")
                nomeGraf3.place(rely=0.5, relx=0.7)

                df_rows = df_teste.to_numpy().tolist()
                for row in df_rows:
                    resultado.insert("", "end", values=row)
                return None

        def clear_arquivo():
            resultado.delete(*resultado.get_children())

    def button_event(self):
        print("Button pressed")

    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()