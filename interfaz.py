import tkinter as tk
from tkinter import ttk, messagebox
from main import ejecutar_algoritmo_genetico

class InterfazGenetico:
    def __init__(self, root):
        self.root = root
        self.root.title("Algoritmo Genético")

        # Variables para almacenar los valores ingresados por el usuario
        self.deltaX_var = tk.DoubleVar()
        self.a_var = tk.DoubleVar()
        self.b_var = tk.DoubleVar()
        self.poblacion_inicial_var = tk.IntVar()
        self.poblacion_maxima_var = tk.IntVar()
        self.tipo_optimizacion_var = tk.StringVar()
        self.ecuacion_var = tk.StringVar()
        self.prob_mutacion_individuo_var = tk.DoubleVar()
        self.prob_mutacion_gen_var = tk.DoubleVar()
        self.num_generaciones_var = tk.IntVar()
        self.mejor_individuo_global = None




        # Llamar a la función de configuración de la interfaz
        self.configurar_interfaz()

    def configurar_interfaz(self):
        # Crear y configurar etiquetas y campos de entrada
        ttk.Label(self.root, text="Delta X:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        ttk.Entry(self.root, textvariable=self.deltaX_var).grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="a:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        ttk.Entry(self.root, textvariable=self.a_var).grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="b:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        ttk.Entry(self.root, textvariable=self.b_var).grid(row=2, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="Población Inicial:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
        ttk.Entry(self.root, textvariable=self.poblacion_inicial_var).grid(row=3, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="Población Máxima:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
        ttk.Entry(self.root, textvariable=self.poblacion_maxima_var).grid(row=4, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="Tipo de Optimización:").grid(row=5, column=0, padx=10, pady=5, sticky="e")
        tipo_optimizacion_combobox = ttk.Combobox(self.root, textvariable=self.tipo_optimizacion_var,
                                                  values=["min", "max"], state="readonly")
        tipo_optimizacion_combobox.grid(row=5, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="Ecuación:").grid(row=6, column=0, padx=10, pady=5, sticky="e")
        ttk.Entry(self.root, textvariable=self.ecuacion_var).grid(row=6, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="Prob. Mutación Individuo:").grid(row=7, column=0, padx=10, pady=5, sticky="e")
        ttk.Entry(self.root, textvariable=self.prob_mutacion_individuo_var).grid(row=7, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="Prob. Mutación Gen:").grid(row=8, column=0, padx=10, pady=5, sticky="e")
        ttk.Entry(self.root, textvariable=self.prob_mutacion_gen_var).grid(row=8, column=1, padx=10, pady=5)
        ttk.Label(self.root, text="Número de Generaciones:").grid(row=9, column=0, padx=10, pady=5, sticky="e")

        ttk.Label(self.root, text="Número de Generaciones:").grid(row=9, column=0, padx=10, pady=5, sticky="e")
        ttk.Entry(self.root, textvariable=self.num_generaciones_var).grid(row=9, column=1, padx=10, pady=5)

        # Crear botón para ejecutar 
        ttk.Button(self.root, text="Ejecutar Algoritmo Genético", command=self.ejecutar_genetico).grid(row=10, column=0, columnspan=2, pady=10)

    def ejecutar_genetico(self):

        # Obtener los valores ingresados por el usuario
        delta_X = self.deltaX_var.get()
        a = self.a_var.get()
        b = self.b_var.get()
        poblacion_inicial = self.poblacion_inicial_var.get()
        poblacion_maxima = self.poblacion_maxima_var.get()
        tipo_optimizacion = self.tipo_optimizacion_var.get()
        ecuacion = self.ecuacion_var.get()
        prob_mutacion_individuo = self.prob_mutacion_individuo_var.get()
        prob_mutacion_gen = self.prob_mutacion_gen_var.get()
        num_generaciones = self.num_generaciones_var.get()
        resultado_genetico = ejecutar_algoritmo_genetico(delta_X, a, b, poblacion_inicial, poblacion_maxima,
                                                        tipo_optimizacion, ecuacion, prob_mutacion_individuo,
                                                        prob_mutacion_gen, num_generaciones)

        self.mejor_individuo_global = resultado_genetico[0]

        messagebox.showinfo("Mejor Individuo Global", f"Mejor Individuo Global:\n{self.mejor_individuo_global}")




if __name__ == "__main__":
    root = tk.Tk()
    interfaz = InterfazGenetico(root)
    root.mainloop()
