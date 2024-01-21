import random
from math import log2
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, cos, lambdify, tan
def ejecutar_algoritmo_genetico(deltaX_value, a_value, b_value, poblacion_inicial_value, poblacion_maxima_deseada_value, tipo_optimizacion_value, valor_ecuacion, valor_prob_mutacion_individuo,
                                valor_prob_mutacion_gen , num_generaciones):


    # Declarar variables globales
    a = 0
    b = 0
    deltaX = 0
    poblacion_inicial = 0
    deltaX_optima = 0
    mejor_individuo_global = None
    tipo_optimizacion = None

    def generar_individuo(n):
        individuo = ''.join(random.choice('01') for _ in range(n))
        return individuo

    def operaciones_basicas(deltaX_value, a_value, b_value, poblacion_inicial_value, poblacion_maxima_deseada_value):
        nonlocal a, b, deltaX, poblacion_inicial, deltaX_optima, poblacion_maxima_deseada, tipo_optimizacion  # Declarar como globales

        # Eliminar la línea de entrada mediante input y utilizar el valor recibido desde la interfaz gráfica
        deltaX = deltaX_value
        a = a_value
        b = b_value
        poblacion_inicial = poblacion_inicial_value
        poblacion_maxima_deseada = poblacion_maxima_deseada_value
        tipo_optimizacion = tipo_optimizacion_value.lower()

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

        return int(n), deltaX_optima, poblacion_maxima_deseada

    def generar_poblacion(n, deltaX_optima, funcion):
        poblacion = []

        # Utilizar el mejor individuo global como parte de la población inicial si está disponible
        if mejor_individuo_global is not None:
            poblacion.append(mejor_individuo_global)

        # Completar la población con individuos generados aleatoriamente
        while len(poblacion) < poblacion_inicial:
            individuo = generar_individuo(n)
            valor_entero = int(individuo, 2)
            x = a + valor_entero * deltaX_optima  # Calcula el valor de x
            x = round(x, 4)
            fx = funcion(x)
            fx = round(fx, 4)
            poblacion.append([individuo, valor_entero, x, fx])

        return poblacion

    def ingresar_ecuacion():
        x = symbols('x')
        ecuacion = valor_ecuacion
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

    def seleccionar_mejores(poblacion, porcentaje):
        # Ordenar la población según la aptitud (f(x))
        if tipo_optimizacion == 'max':
            poblacion_ordenada = sorted(poblacion, key=lambda ind: ind[3], reverse=True)
        elif tipo_optimizacion == 'min':
            poblacion_ordenada = sorted(poblacion, key=lambda ind: ind[3], reverse=False)
        else:
            print("Error: Tipo de optimización no válido.")
            exit()

        # Si la población es muy pequeña, cruzar los individuos disponibles
        if len(poblacion) <= 2:
            return list(zip(poblacion, poblacion[::-1]))  # Cruza el único par posible

        # Seleccionar el porcentaje de mejores individuos
        cantidad_mejores = max(int(len(poblacion) * porcentaje), 2)

        if cantidad_mejores % 2 != 0:
            cantidad_mejores += 1  # Asegurar un número par de mejores individuos

        mejores = poblacion_ordenada[:cantidad_mejores]
        demas = poblacion_ordenada[cantidad_mejores:]

        # Asegurarse de tener al menos dos individuos para formar una pareja
        if len(mejores) < 2:
            mejores.append(demas.pop(0))

        # Formar parejas a partir de los mejores y los demás individuos
        parejas = list(zip(mejores[:len(mejores) // 2], demas))

        return parejas

    def cruzar_parejas(parejas, deltaX_optima, funcion):
        hijos = []

        for pareja in parejas:
            # Obtener los individuos de la pareja
            individuo1, individuo2 = pareja[0][0], pareja[1][0]

            # Seleccionar aleatoriamente el punto de cruce
            punto_cruce = random.randint(1, len(individuo1) - 1)

            # Realizar la cruza
            hijo1_bin = individuo1[:punto_cruce] + individuo2[punto_cruce:]
            hijo2_bin = individuo2[:punto_cruce] + individuo1[punto_cruce:]

            # Convertir la cadena de bits a valor entero para ambos hijos
            hijo1_valor = int(hijo1_bin, 2)
            hijo2_valor = int(hijo2_bin, 2)

            # Calcular el valor de x para ambos hijos
            hijo1_x = a + hijo1_valor * deltaX_optima
            hijo2_x = a + hijo2_valor * deltaX_optima

            # Redondear el valor de x a 4 decimales
            hijo1_x = round(hijo1_x, 4)
            hijo2_x = round(hijo2_x, 4)

            # Calcular el valor de f(x) para ambos hijos
            hijo1_fx = funcion(hijo1_x)
            hijo2_fx = funcion(hijo2_x)

            # Agregar los hijos a la lista de hijos
            hijos.extend([[hijo1_bin, hijo1_valor, hijo1_x, hijo1_fx],
                          [hijo2_bin, hijo2_valor, hijo2_x, hijo2_fx]])

            # Imprimir el punto de cruce (opcional)
            print(f"Punto de cruce para pareja {pareja[0][0]}, {pareja[1][0]}: {punto_cruce + 1}")

        return hijos

    def calcular_promedio(poblacion):
        return sum(ind[3] for ind in poblacion) / len(poblacion)

    # Algoritmo genético
    n, deltaX_optima, poblacion_maxima_deseada = operaciones_basicas(deltaX_value, a_value, b_value,
                                                                     poblacion_inicial_value,
                                                                     poblacion_maxima_deseada_value)

    funcion = ingresar_ecuacion()  # Declarar la función aquí

    # Número total de generaciones
    num_generaciones = num_generaciones

    # Lista para almacenar todos los individuos de cada generación
    todos_los_individuos = []

    def mostrar_parejas(parejas):
        print("Parejas seleccionadas:")
        for i, pareja in enumerate(parejas, start=1):
            print(f"Pareja{i}: {pareja[0][0]} {pareja[1][0]}")

    def mutar_individuo(individuo, prob_mutacion_gen, deltaX_optima, funcion, a):
        bits_mutados = ""
        for bit in individuo[0]:  # Seleccionar solo la cadena de bits
            # Aplicar mutación al bit con probabilidad prob_mutacion_gen
            if random.random() < prob_mutacion_gen:
                # Negar el bit y agregarlo a la cadena mutado
                bits_mutados += '0' if bit == '1' else '1'
            else:
                # Agregar el bit sin mutar a la cadena mutado
                bits_mutados += bit

        # Convertir la cadena de bits mutada a valor entero
        valor_entero_mutado = int(bits_mutados, 2)
        # Calcular el nuevo valor de x usando el valor entero y deltaX_optima
        x_mutado = a + valor_entero_mutado * deltaX_optima
        # Redondear x a 4 decimales
        x_mutado = round(x_mutado, 4)
        # Calcular el nuevo valor de f(x) para el x mutado
        fx_mutado = funcion(x_mutado)

        # Retornar una lista que contenga la cadena de bits y los demás valores recalculados
        return [bits_mutados, valor_entero_mutado, x_mutado, fx_mutado]

    def eliminar_repetidos(poblacion):
        individuos_unicos = set()
        poblacion_sin_repetidos = []

        for individuo in poblacion:
            # Convierte la cadena de bits en una tupla para que sea hashable
            individuo_tupla = tuple(individuo[0])
            if individuo_tupla not in individuos_unicos:
                individuos_unicos.add(individuo_tupla)
                poblacion_sin_repetidos.append(individuo)

        return poblacion_sin_repetidos

    def eliminar_excedentes_aleatoriamente(poblacion, poblacion_maxima_deseada, tipo_optimizacion):
        # Identificar el mejor individuo según el tipo de optimización
        if tipo_optimizacion == 'max':
            mejor_individuo = max(poblacion, key=lambda ind: ind[3])
        elif tipo_optimizacion == 'min':
            mejor_individuo = min(poblacion, key=lambda ind: ind[3])
        else:
            raise ValueError("Tipo de optimización no válido.")

        # Asegurarse de que el mejor individuo no se elimine
        poblacion.remove(mejor_individuo)

        # Eliminar individuos al azar hasta alcanzar el tamaño deseado
        while len(poblacion) > poblacion_maxima_deseada:
            individuo_a_eliminar = random.choice(poblacion)
            poblacion.remove(individuo_a_eliminar)

        # Añadir el mejor individuo de vuelta a la población
        poblacion.append(mejor_individuo)

        return poblacion

    prob_mutacion_individuo = valor_prob_mutacion_individuo
    prob_mutacion_gen = valor_prob_mutacion_gen
    # Listas para almacenar datos de cada generación

    mejores_fitness = []
    peores_fitness = []
    promedio_fitness = []
    hijos_mutados = []

    poblacion_resultados = generar_poblacion(n, deltaX_optima, funcion)

    for generacion in range(num_generaciones):
        poblacion_resultados = eliminar_repetidos(poblacion_resultados)

        # Encontrar el mejor individuo de la generación actual
        mejor_individuo_generacion = encontrar_mejor_individuo(poblacion_resultados)

        # Aplicar selección de los mejores individuos
        porcentaje_seleccion = 0.3
        parejas_seleccionadas = seleccionar_mejores(poblacion_resultados, porcentaje_seleccion)

        # Mostrar parejas seleccionadas
        mostrar_parejas(parejas_seleccionadas)

        # Aplicar cruza a las parejas seleccionadas
        hijos_cruzados = cruzar_parejas(parejas_seleccionadas, deltaX_optima, funcion)

        # Imprimir los resultados de la cruza
        print("Hijos resultantes de la cruza:")
        for i, hijo in enumerate(hijos_cruzados, start=1):
            print(f"Hijo{i}: {hijo}")

        # Aplicar mutación solo a los hijos
        numero_aleatorio = random.uniform(0.0, 1.0)
        numero_aleatorio2 = random.uniform(0.0, 1.0)
        if numero_aleatorio <= prob_mutacion_individuo:
            if prob_mutacion_gen <= numero_aleatorio2:
                hijos_mutados = [mutar_individuo(hijo, prob_mutacion_gen, deltaX_optima, funcion, a) for hijo in
                                 hijos_cruzados]

        poblacion_resultados.extend(hijos_mutados)

        # Actualizar el mejor individuo global si es necesario
        if mejor_individuo_global is None or mejor_individuo_generacion[3] < mejor_individuo_global[3]:
            mejor_individuo_global = mejor_individuo_generacion

        if len(poblacion_resultados) > poblacion_maxima_deseada:
            poblacion_resultados = eliminar_excedentes_aleatoriamente(poblacion_resultados, poblacion_maxima_deseada,
                                                                      tipo_optimizacion)

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

        if len(poblacion_resultados) > poblacion_maxima_deseada:
            # Eliminación aleatoria asegurando mantener al mejor individuo
            if tipo_optimizacion == 'max':
                poblacion_resultados.sort(key=lambda ind: ind[3], reverse=True)
            elif tipo_optimizacion == 'min':
                poblacion_resultados.sort(key=lambda ind: ind[3], reverse=False)
            else:
                print("Error: Tipo de optimización no válido.")
                exit()  # Ordenar por aptitud de mayor a menor
            poblacion_resultados = poblacion_resultados[:poblacion_maxima_deseada]
        if tipo_optimizacion == 'max':
            peor_individuo_generacion = min(poblacion_resultados, key=lambda ind: ind[3])
        elif tipo_optimizacion == 'min':
            peor_individuo_generacion = max(poblacion_resultados, key=lambda ind: ind[3])
        else:
            print("Error: Tipo de optimización no válido.")
            exit()
        if tipo_optimizacion == 'max':
            destacado = max(poblacion_resultados, key=lambda ind: ind[3])
        elif tipo_optimizacion == 'min':
            destacado = min(poblacion_resultados, key=lambda ind: ind[3])

        valores_x = [individuo[2] for individuo in poblacion_resultados]  # Valores de x
        valores_fx = [individuo[3] for individuo in poblacion_resultados]  # Valores de f(x)
        # Calcular el peor de la generación actual
        peor_individuo_generacion = peor_individuo_generacion
        # Almacenar todos los individuos de la generación actual
        todos_los_individuos.extend(poblacion_resultados)

        # Almacenar datos para graficar
        mejores_fitness.append(mejor_individuo_generacion[3])
        peores_fitness.append(peor_individuo_generacion[3])
        promedio_fitness.append(calcular_promedio(poblacion_resultados))

    print("\nPoblación Final:")
    print("Individuo, Valor Entero, x, f(x)")
    for individuo in poblacion_resultados:
        print(f"{individuo[0]}, {individuo[1]}, {individuo[2]}, {individuo[3]}")
    print(len(poblacion_resultados))

    # Graficar resultados
    generaciones = list(range(1, num_generaciones + 1))

    plt.plot(generaciones, mejores_fitness, label='Mejor')
    plt.plot(generaciones, peores_fitness, label='Peor')
    plt.plot(generaciones, promedio_fitness, label='Promedio')
    plt.xlabel('Generación')
    plt.ylabel('Valores')
    plt.legend()

    # Cambiar colores en la gráfica si es maximización
    if tipo_optimizacion == 'max':
        plt.gca().get_lines()[0].set_color('blue')  # Mejor en verde
        plt.gca().get_lines()[1].set_color('red')  # Peor en rojo
        plt.gca().get_lines()[2].set_color('green')  # Promedio en azul

    if tipo_optimizacion == 'min':
        plt.gca().get_lines()[0].set_color('blue')  # Mejor en verde
        plt.gca().get_lines()[1].set_color('red')  # Peor en rojo
        plt.gca().get_lines()[2].set_color('green')  # Promedio en azul
    plt.title(f'Evolucion de fitness de las {num_generaciones} generaciones')
    plt.show()

    x_vals = np.linspace(a, b, 400)
    fx_vals = funcion(x_vals)

    # Segunda gráfica: Dispersión de f(x) para la última generación y la funció1n objetivo
    plt.figure(figsize=(10, 6))
    # Traza la función objetivo
    plt.plot(x_vals, fx_vals, label='f(x) original', color='black')

    # Traza la dispersión de los individuos
    plt.scatter(valores_x, valores_fx, alpha=0.6, label='Individuos')

    # Destacar el mejor individuo (o peor, dependiendo de la optimización)
    plt.scatter(destacado[2], destacado[3], color='red', s=100, label='Mejor')

    plt.title(f'Dispersión de f(x) de las {num_generaciones} generaciones')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

    # Imprimir todos los individuos de cada generación al final
    print("Todos los individuos de cada generación:")
    print("Generación, Individuo, Valor Entero, x, f(x)")
    for generacion, individuo in enumerate(todos_los_individuos, start=1):
        print(f"{generacion}, {individuo[0]}, {individuo[1]}, {individuo[2]}, {individuo[3]}")

    # Imprimir el mejor individuo global al final
    print(f"Mejor individuo global: {mejor_individuo_global}")
    print(f"La poblacion final fue {len(poblacion_resultados)}")
