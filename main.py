import random
from math import log2
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, cos, lambdify

# Declarar variables globales
a = 0
b = 0
deltaX = 0
poblacion_maxima = 0
deltaX_optima = 0
mejor_individuo_global = None
tipo_optimizacion = None


def generar_individuo(n):
    individuo = ''.join(random.choice('01') for _ in range(n))
    return individuo


def operaciones_basicas():
    global a, b, deltaX, poblacion_maxima, deltaX_optima, tipo_optimizacion  # Declarar como globales
    deltaX = float(input("Ingrese el valor de deltaX: "))
    a = float(input("Ingrese el valor de a: "))
    b = float(input("Ingrese el valor de b: "))
    poblacion_maxima = int(input("Ingrese el valor de la población máxima: "))
    tipo_optimizacion = input("Ingrese 'max' para maximización o 'min' para minimización: ").lower()

    if deltaX <= 0 or a >= b:
        print("Error: Verifique que deltaX sea positivo y que 'a' sea menor que 'b'.")
        exit()

    rango = b - a
    saltos = rango / deltaX
    puntos = saltos + 1
    n = int(log2(puntos))

    while not (2 ** (n - 1) < puntos <= 2 ** n):
        if puntos > 2 ** n:
            n += 1
        elif puntos < 2 ** (n - 1):
            n -= 1
    deltaX2 = rango / (2 ** n - 1)

    if deltaX < deltaX2:
        deltaX_optima = deltaX
    else:
        deltaX_optima = deltaX2

    return int(n), deltaX_optima


def generar_poblacion(n, deltaX_optima, funcion):
    poblacion = []

    # Utilizar el mejor individuo global como parte de la población inicial si está disponible
    if mejor_individuo_global is not None:
        poblacion.append(mejor_individuo_global)

    # Completar la población con individuos generados aleatoriamente
    while len(poblacion) < poblacion_maxima:
        individuo = generar_individuo(n)
        valor_entero = int(individuo, 2)
        x = a + valor_entero * deltaX_optima  # Calcula el valor de x
        fx = funcion(x)  # Calcula el valor de f(x)
        poblacion.append([individuo, valor_entero, x, fx])

    return poblacion


def ingresar_ecuacion():
    x = symbols('x')
    ecuacion = (x**3)*cos(x)
    return lambdify(x, ecuacion, 'numpy')


def encontrar_mejor_individuo(poblacion):
    # Encontrar el mejor individuo según la función de aptitud (fx)
    if tipo_optimizacion == 'max':
        mejor_individuo = max(poblacion, key=lambda ind: ind[3])
    elif tipo_optimizacion == 'min':
        mejor_individuo = min(poblacion, key=lambda ind: ind[3])
    else:
        print("Error: Tipo de optimización no válido.")
        exit()

    return mejor_individuo


def calcular_promedio(poblacion):
    return sum(ind[3] for ind in poblacion) / len(poblacion)


# Algoritmo genético
n, deltaX_optima = operaciones_basicas()
funcion = ingresar_ecuacion()  # Declarar la función aquí

# Número total de generaciones que deseas
num_generaciones = 5

# Lista para almacenar todos los individuos de cada generación
todos_los_individuos = []

# Listas para almacenar datos de cada generación
mejores_fitness = []
peores_fitness = []
promedio_fitness = []

for generacion in range(num_generaciones):
    poblacion_resultados = generar_poblacion(n, deltaX_optima, funcion)

    # Encontrar el mejor individuo de la generación actual
    mejor_individuo_generacion = encontrar_mejor_individuo(poblacion_resultados)

    # Actualizar el mejor individuo global si es necesario
    if mejor_individuo_global is None or mejor_individuo_generacion[3] < mejor_individuo_global[3]:
        mejor_individuo_global = mejor_individuo_generacion

    # Almacenar todos los individuos de la generación actual
    todos_los_individuos.extend(poblacion_resultados)

    # Encontrar el mejor individuo de la generación actual
    if tipo_optimizacion == 'max':
        mejor_individuo_generacion = max(poblacion_resultados, key=lambda ind: ind[3])
        peor_individuo_generacion = min(poblacion_resultados, key=lambda ind: ind[3])
    elif tipo_optimizacion == 'min':
        mejor_individuo_generacion = min(poblacion_resultados, key=lambda ind: ind[3])
        peor_individuo_generacion = max(poblacion_resultados, key=lambda ind: ind[3])
    else:
        print("Error: Tipo de optimización no válido.")
        exit()

    # Actualizar el mejor individuo global si es necesario
    if mejor_individuo_global is None or mejor_individuo_generacion[3] > mejor_individuo_global[3]:
        mejor_individuo_global = mejor_individuo_generacion

    # Calcular el peor de la generación actual
    peor_individuo_generacion = peor_individuo_generacion
    # Almacenar todos los individuos de la generación actual
    todos_los_individuos.extend(poblacion_resultados)

    # Almacenar datos para graficar
    mejores_fitness.append(mejor_individuo_generacion[3])
    peores_fitness.append(peor_individuo_generacion[3])
    promedio_fitness.append(calcular_promedio(poblacion_resultados))

# Graficar resultados
generaciones = list(range(1, num_generaciones + 1))

plt.plot(generaciones, mejores_fitness, label='Mejor')
plt.plot(generaciones, peores_fitness, label='Peor')
plt.plot(generaciones, promedio_fitness, label='Promedio')
plt.xlabel('Generación')
plt.ylabel('f(x)')

# Cambiar colores en la gráfica si es maximización
if tipo_optimizacion == 'max':
    plt.gca().get_lines()[0].set_color('green')  # Mejor en verde
    plt.gca().get_lines()[1].set_color('red')  # Peor en rojo
    plt.gca().get_lines()[2].set_color('blue')  # Promedio en azul

plt.legend()
plt.title('Mejor, Peor y Promedio de cada Generación')
plt.show()

# Imprimir todos los individuos de cada generación al final
print("Todos los individuos de cada generación:")
print("Generación, Individuo, Valor Entero, x, f(x)")
for generacion, individuo in enumerate(todos_los_individuos, start=1):
    print(f"{generacion}, {individuo[0]}, {individuo[1]}, {individuo[2]}, {individuo[3]}")

# Imprimir el mejor individuo global al final
print(f"Mejor individuo global: {mejor_individuo_global}")
