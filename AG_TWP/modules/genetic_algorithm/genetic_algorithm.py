import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================
K_FOLDS = 5  # Número de folds para validación cruzada
LAMBDA_PENALTY = 0.5  # Reducido de 2.65 a 1.0 (menos penalización)

# ============================================================================
# DEFINICIÓN DE ESTRUCTURAS (DEAP) - SE EJECUTA UNA SOLA VEZ
# ============================================================================
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

# ============================================================================
# TOOLBOX GLOBAL
# ============================================================================
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# ============================================================================
# FUNCIÓN DE FITNESS MEJORADA
# ============================================================================
def evaluate_features(individual, X_data, Y_targets, total_features, lambda_penalty, k_folds):
    """
    Calcula el fitness de un individuo usando validación cruzada con penalización por complejidad.
    
    Args:
        individual: Vector binario (0/1) indicando qué características usar
        X_data: Matriz de características (n_samples, n_features)
        Y_targets: Vector de etiquetas (n_samples,)
        total_features: Número total de características disponibles
        lambda_penalty: Peso de la penalización por complejidad
        k_folds: Número de folds para validación cruzada
        
    Returns:
        tuple: (fitness_score,) - Mayor es mejor
    """
    
    # 1. Extraer índices de características seleccionadas
    selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
    
    # Penalización si no se selecciona ninguna característica
    if not selected_indices:
        return (0.0,)
    
    n_selected = len(selected_indices)
    X_reduced = X_data[:, selected_indices]
    
    # 2. Configurar clasificador LDA
    clf = LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')
    
    # 3. Calcular accuracy con validación cruzada
    try:
        scores = cross_val_score(clf, X_reduced, Y_targets, 
                                cv=k_folds, scoring='accuracy')
        accuracy_cv = np.mean(scores) * 100  # Convertir a porcentaje
    except Exception as e:
        # Si falla la validación cruzada (ej: muy pocas características)
        return (0.0,)
    
    # 4. Calcular penalización por complejidad (sparsity)
    sparsity_ratio = n_selected / total_features
    penalty = lambda_penalty * sparsity_ratio * 100
    
    # 5. Fitness final = Accuracy - Penalización
    fitness_score = accuracy_cv - penalty
    
    return (fitness_score,)

# ============================================================================
# FUNCIÓN PRINCIPAL CON EARLY STOPPING
# ============================================================================
def run_genetic_algorithm(X_twp, Y_targets, pop_size=100, num_generations=100, 
                         lambda_penalty=None, early_stopping_patience=15):
    """
    Ejecuta el algoritmo genético para selección de características.
    
    Args:
        X_twp: Matriz de características TWP (n_samples, n_features)
        Y_targets: Vector de etiquetas (n_samples,)
        pop_size: Tamaño de la población
        num_generations: Número máximo de generaciones
        lambda_penalty: Peso de penalización (None = usar global)
        early_stopping_patience: Detener si no hay mejora en N generaciones
        
    Returns:
        best_individual: Mejor solución encontrada
        log: Registro de estadísticas por generación
    """
    
    # Usar lambda global si no se especifica
    if lambda_penalty is None:
        lambda_penalty = LAMBDA_PENALTY
    
    # Calcular número de características dinámicamente
    current_n_features = X_twp.shape[1]
    
    # Registrar el tamaño correcto del individuo
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                    toolbox.attr_bool, n=current_n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Registrar función de evaluación con parámetros específicos
    toolbox.register("evaluate", evaluate_features, 
                    X_data=X_twp, 
                    Y_targets=Y_targets, 
                    total_features=current_n_features, 
                    lambda_penalty=lambda_penalty,
                    k_folds=K_FOLDS)
    
    # Crear población inicial
    pop = toolbox.population(n=pop_size)
    
    # Configurar estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    
    # Hall of Fame (guardar mejores individuos)
    hof = tools.HallOfFame(5)  # Guardar top 5
    
    # ========================================================================
    # ALGORITMO CON EARLY STOPPING
    # ========================================================================
    best_fitness_history = []
    generations_without_improvement = 0
    
    for gen in range(num_generations):
        # Evaluar población
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Actualizar Hall of Fame
        hof.update(pop)
        
        # Registrar estadísticas
        record = stats.compile(pop)
        best_fitness_history.append(record['max'])
        
        # Imprimir progreso cada 10 generaciones
        if gen % 10 == 0 or gen == num_generations - 1:
            print(f"Gen {gen:3d} | Max: {record['max']:7.3f} | "
                  f"Avg: {record['avg']:7.3f} | Std: {record['std']:6.3f}")
        
        # Early stopping
        if gen > 0:
            if best_fitness_history[-1] <= best_fitness_history[-2]:
                generations_without_improvement += 1
            else:
                generations_without_improvement = 0
        
        if generations_without_improvement >= early_stopping_patience:
            print(f"⚠️  Early stopping en generación {gen} "
                  f"(sin mejora por {early_stopping_patience} generaciones)")
            break
        
        # Última generación
        if gen == num_generations - 1:
            break
        
        # Selección
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Cruce
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # Probabilidad de cruce
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutación
        for mutant in offspring:
            if random.random() < 0.3:  # Probabilidad de mutación
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Reemplazar población
        pop[:] = offspring
    
    # ========================================================================
    # RESULTADOS FINALES
    # ========================================================================
    best_individual = hof[0]
    
    # Crear log compatible con DEAP
    log = tools.Logbook()
    log.record(gen=len(best_fitness_history), 
              max=best_fitness_history[-1],
              best_individual=best_individual)
    
    return best_individual, log

# ============================================================================
# FUNCIÓN AUXILIAR: GRID SEARCH PARA LAMBDA
# ============================================================================
def optimize_lambda(X_twp, Y_targets, lambda_values=[0.1, 0.5, 1.0, 2.0, 3.0],
                   pop_size=50, num_generations=30):
    """
    Encuentra el mejor valor de lambda mediante grid search.
    
    Args:
        X_twp: Matriz de características
        Y_targets: Etiquetas
        lambda_values: Lista de valores de lambda a probar
        pop_size: Tamaño de población (reducido para rapidez)
        num_generations: Generaciones (reducido para rapidez)
        
    Returns:
        best_lambda: Mejor valor de lambda encontrado
        results: Diccionario con resultados de cada lambda
    """
    
    print("🔍 Optimizando valor de lambda...")
    results = {}
    
    for lam in lambda_values:
        print(f"\n--- Probando λ = {lam} ---")
        best_ind, _ = run_genetic_algorithm(
            X_twp, Y_targets, 
            pop_size=pop_size, 
            num_generations=num_generations,
            lambda_penalty=lam,
            early_stopping_patience=10
        )
        
        selected = [i for i, bit in enumerate(best_ind) if bit == 1]
        fitness = best_ind.fitness.values[0]
        
        results[lam] = {
            'fitness': fitness,
            'n_features': len(selected),
            'individual': best_ind
        }
        
        print(f"Fitness: {fitness:.3f} | Features: {len(selected)}")
    
    # Encontrar mejor lambda
    best_lambda = max(results.keys(), key=lambda k: results[k]['fitness'])
    
    print(f"\n✅ Mejor λ = {best_lambda} (Fitness: {results[best_lambda]['fitness']:.3f})")
    
    return best_lambda, results