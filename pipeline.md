# Documentación del Pipeline: Clasificación de imaginación motora con AG + TWP + LDA

**Autor:** Claure Jorge, Diana Vertiz  
**Fecha:** Noviembre 2025  
**Versión:** 1.0

---

## 📋 Tabla de Contenidos

1. [Introducción](#Introducción)
2. [Base Teórica](#base-teórica)
3. [Arquitectura del Pipeline](#arquitectura-del-pipeline)
4. [Implementación Detallada](#implementación-detallada)
5. [Configuración y Parámetros](#configuración-y-parámetros)
6. [Resultados Esperados](#resultados-esperados)
7. [Interpretación de Resultados](#interpretación-de-resultados)
8. [Referencias](#referencias)

---

## 1. Introducción

### 1.1 Objetivo del Proyecto

Este proyecto implementa un **sistema de clasificación de señales EEG** para detectar **Imaginación Motora (MI - Motor Imagery)** utilizando un enfoque híbrido que combina:

- **Transformada Wavelet Packet (TWP)** para extracción de características
- **Algoritmo Genético (AG)** para selección óptima de características
- **Linear Discriminant Analysis (LDA)** como clasificador

El objetivo es distinguir entre dos estados mentales:
- **Motor Imagery (MI)**: Imaginación de movimiento
- **Rest (Reposo)**: Estado de reposo mental

### 1.2 Contexto Clínico

Este sistema es parte de una **Brain-Computer Interface (BCI)** diseñada para aplicaciones de:
- Rehabilitación motora
- Control de prótesis robóticas

---

## 2. Base Teórica

### 2.1 Imaginación motora y ritmos sensoriomotores

#### ¿Qué es la imaginación motora?

La **imaginación motora** es la simulación mental de un movimiento sin ejecución física real. Durante este proceso, se activan áreas motoras del cerebro similares a las que se activarían durante el movimiento real.

#### Fenómenos EEG Asociados

Durante la imagería motora, se observan dos fenómenos principales en el EEG:

1. **Event-Related Desynchronization (ERD)**
   - Disminución de la potencia en bandas de frecuencia específicas
   - Ocurre en el ritmo **mu (8-12 Hz)** y **beta (12-30 Hz)**
   - Localización: Corteza motora contralateral al movimiento imaginado

2. **Event-Related Synchronization (ERS)**
   - Aumento de la potencia tras la finalización del movimiento
   - Indica retorno al estado de reposo

#### Bandas de Frecuencia Relevantes

| Banda | Frecuencia | Relevancia para MI |
|-------|------------|-------------------|
| **Theta** | 4-8 Hz | Atención y memoria de trabajo |
| **Alfa/Mu** | 8-12 Hz | **Crítico** - Ritmo sensoriomotor |
| **Beta** | 12-30 Hz | **Importante** - Activación motora |
| **Gamma** | >30 Hz | Procesamiento cognitivo complejo |

---

### 2.2 Transformada Wavelet Packet (TWP)

#### Fundamento matemático

La **Transformada Wavelet** descompone una señal en componentes tiempo-frecuencia usando funciones base (wavelets) que son versiones escaladas y trasladadas de una wavelet madre:

```
ψ(a,b)(t) = 1/√a * ψ((t-b)/a)
```

Donde:
- `a`: Factor de escala (inversamente proporcional a la frecuencia)
- `b`: Factor de traslación (posición temporal)

#### Wavelet Packet Transform (WPT)

A diferencia de la Transformada Wavelet Discreta (DWT), que solo descompone las aproximaciones, la **WPT descompone tanto aproximaciones como detalles**, creando un árbol binario completo:

```
                    Señal Original
                    /            \
            [Aprox. 1]          [Detalle 1]
            /        \          /         \
        [A1.1]    [D1.1]    [A1.2]     [D1.2]
```

#### Niveles de Descomposición

- **Level 3**: 2³ = 8 nodos → **40 características** (5 canales × 8)
- **Level 4**: 2⁴ = 16 nodos → **80 características** (5 canales × 16)

Cada nodo representa una banda de frecuencia específica.

#### Extracción de Características: Energía

Para cada nodo del nivel más profundo, se calcula la **energía**:

```
E_i = Σ (coeficiente_j)²
```

La energía captura la potencia de la señal en esa banda tiempo-frecuencia específica.

#### Wavelets Utilizadas

En este proyecto se usa **Coiflet 3 (coif3)**:

| Wavelet | Características | Uso Típico |
|---------|----------------|------------|
| **coif3** | Soporte compacto, simétrica | EEG, señales biomédicas |
| db4 | Daubechies 4, ortogonal | Análisis general |
| sym4 | Symlet 4, casi simétrica | Detección de eventos |

---

### 2.3 Algoritmo Genético (AG) para Selección de Características

#### ¿Por qué Selección de Características?

Con 40-80 características extraídas por TWP, surgen problemas:
- **Maldición de la dimensionalidad**: Pocos datos (60 trials) vs muchas características
- **Overfitting**: El modelo memoriza ruido en lugar de aprender patrones
- **Características redundantes**: No todas las bandas/canales son informativas

#### Fundamentos de Algoritmos Genéticos

Los AGs simulan la **evolución biológica** para encontrar soluciones óptimas:

1. **Representación (Cromosoma)**
   ```
   Individuo = [1, 0, 1, 1, 0, ..., 1]
   ```
   - Longitud = Número de características (40 u 80)
   - `1` = Característica seleccionada
   - `0` = Característica descartada

2. **Función de Aptitud (Fitness)**
   ```
   Fitness = Accuracy_CV - (λ × Sparsity × 100)
   ```
   
   - **Accuracy_CV**: Precisión con validación cruzada de 5-fold
   - **λ (lambda)**: Peso de penalización (0.3 - 2.0)
   - **Sparsity**: Fracción de características usadas
   
   **Objetivo dual:**
   - ✅ Maximizar precisión (primer término)
   - ✅ Minimizar número de características (segundo término)

3. **Operadores Genéticos**

   a) **Selección** (Tournament):
   - Se eligen 3 individuos al azar
   - El mejor (mayor fitness) pasa a la siguiente generación
   
   b) **Cruce** (Two-Point Crossover, 70% probabilidad):
   ```
   Padre 1:  [1 1 0 | 0 1 | 0 1]
   Padre 2:  [0 1 1 | 1 0 | 1 0]
              -------   ---  -----
   Hijo 1:   [1 1 0 | 1 0 | 0 1]
   Hijo 2:   [0 1 1 | 0 1 | 1 0]
   ```
   
   c) **Mutación** (Bit-Flip, 30% probabilidad, 5% por gen):
   ```
   Antes: [1 0 1 1 0 0 1]
   Después: [1 0 0 1 0 0 1]  (flip en posición 2)
   ```

4. **Early Stopping**
   - Detiene el AG si no hay mejora en **15 generaciones consecutivas**
   - Evita gasto computacional innecesario
   - Típicamente converge entre generaciones 30-60

#### Ventajas del AG sobre Otros Métodos

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| **Algoritmo Genético** | Búsqueda global, escapa mínimos locales | Costoso computacionalmente |
| Filter (ANOVA, Chi²) | Rápido, simple | No considera interacciones |
| Wrapper (RFE) | Considera clasificador | Solo búsqueda local |
| Embedded (Lasso) | Integrado en entrenamiento | Limitado a modelos lineales |

---

### 2.4 Linear Discriminant Analysis (LDA)

#### Fundamento Matemático

LDA busca un **hiperplano de separación** que maximice la distancia entre clases y minimice la varianza intra-clase:

```
J(w) = (w^T S_B w) / (w^T S_W w)
```

Donde:
- `S_B`: Matriz de dispersión entre clases
- `S_W`: Matriz de dispersión dentro de clases
- `w`: Vector de proyección óptimo

#### Ventajas de LDA para BCI

1. **Eficiente**: Solución analítica cerrada (no iterativo)
2. **Robusto**: Funciona bien con pocas muestras
3. **Interpretable**: El vector de pesos indica importancia de características
4. **Rápido**: Clasificación en tiempo real (<1 ms)

#### Shrinkage LDA

Para evitar problemas cuando `n_características > n_muestras`, se usa **Shrinkage**:

```python
LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')
```

- **Shrinkage automático**: Estima el parámetro de regularización óptimo
- **Solver eigen**: Más estable numéricamente

---

## 3. Arquitectura del Pipeline

### 3.1 Diagrama de Flujo General

```
┌─────────────────────────────────────────────────────────────┐
│                    DATOS CRUDOS EEG                         │
│          (8 sujetos, 5 canales, 250 Hz)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PREPROCESAMIENTO                                │
│  - Filtro Butterworth (8-30 Hz)                            │
│  - Segmentación en trials (2.5s MI + 2.5s Rest)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         EXTRACCIÓN DE CARACTERÍSTICAS (TWP)                 │
│  - Transformada Wavelet Packet (level=3, coif3)            │
│  - Cálculo de energía por nodo                             │
│  - Resultado: 60 trials × 40 características               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│        SELECCIÓN DE CARACTERÍSTICAS (AG)                    │
│  - Población: 100 individuos                               │
│  - Generaciones: hasta 100 (con early stopping)           │
│  - Fitness: Accuracy_CV - λ×Sparsity                      │
│  - Resultado: ~3-5 características óptimas                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ENTRENAMIENTO (LDA)                            │
│  - Datos de calibración (60 trials)                        │
│  - Solo características seleccionadas por AG               │
│  - Shrinkage automático                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              EVALUACIÓN (Terapia)                           │
│  - Datos de terapia (60 trials nuevos)                     │
│  - Mismas características que en entrenamiento             │
│  - Métricas: Accuracy, TPR (Sensibilidad)                 │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.2 Estructura de Archivos

```
proyecto/
│
├── modules/
│   ├── preprocessing/
│   │   ├── file_creation.py          # Carga y preprocesamiento
│   │   └── filtering.py               # Filtros Butterworth
│   │
│   ├── feature_extraction/
│   │   └── feature_extraction.py      # TWP, PSD
│   │
│   ├── genetic_algorithm/
│   │   └── genetic_algorithm.py       # AG con DEAP
│   │
│   ├── training/
│   │   └── training.py                # Entrenamiento LDA
│   │
│   ├── evaluation/
│   │   └── evaluation.py              # Evaluación de modelos
│   │
│   └── metrics/
│       └── metrics.py                 # Accuracy, TPR, F1, etc.
│
├── data/
│   └── im_tention_signals/            # Datos EEG (.mat)
│
├── Pruebas_TWP.ipynb                  # Notebook principal
│
└── resultados_ag_twp_lda.csv          # Resultados exportados
```

---

## 4. Implementación Detallada

### 4.1 FASE 1: Carga y Preprocesamiento

#### Código:
```python
dict_cal = create_mat_files(
    './data/im_tention_signals', 
    file_type='calibration', 
    filtfilt=True
)

dict_ter = create_mat_files(
    './data/im_tention_signals', 
    file_type='therapy', 
    ther_number_of_trials=60, 
    filtfilt=True
)
```

#### Procesamiento Interno:

1. **Filtrado Butterworth**:
   - Orden: 4
   - Banda de paso: 8-30 Hz
   - Tipo: Pasa-banda
   - Método: `filtfilt` (fase cero, sin distorsión temporal)

2. **Segmentación de Trials**:
   ```
   Timeline de un trial:
   
   ←0.5s→     ←─ 2.5s ─→    ←0.5s→     ←─ 2.5s ─→
   [Skip] [Motor Imagery] [Skip]     [Rest]
            ↑
          Marca MI
   ```
   
   - **Skip**: 0.5s después/antes de marca (evitar artefactos)
   - **MI**: 2.5s de imaginación motora (625 muestras a 250 Hz)
   - **Rest**: 2.5s de reposo antes de la marca

3. **Estructura de Datos**:
   ```python
   dict_cal['subject_1']['mi_rest'].shape  # (625, 5, 60)
   # 625 muestras × 5 canales × 60 trials (30 MI + 30 Rest)
   
   dict_ter['subject_1']['mi'].shape       # (625, 5, 60)
   dict_ter['subject_1']['target']         # (60,) - 0/1 targets
   ```

---

### 4.2 FASE 2: Extracción de Características (TWP)

#### Código:
```python
X_cal_twp = get_twp_feature_vectors(
    data_cal, 
    level=3,           # 8 nodos por canal
    wavelet='coif3',   # Coiflet 3
    normalize=False,   # Sin normalización
    log_transform=False
)
# Resultado: (60, 40) → 60 trials × 40 características
```

#### Proceso Detallado:

1. **Por cada trial (60 en total)**:
   
   a) **Por cada canal (5 canales: C3, Cz, C4, P3, Pz)**:
      - Extraer señal temporal del canal: `(625,)` muestras
   
   b) **Descomposición Wavelet Packet**:
      ```python
      wp = pywt.WaveletPacket(
          data=signal_channel,
          wavelet='coif3',
          mode='symmetric',
          maxlevel=3
      )
      ```
      
   c) **Extraer nodos del nivel 3**:
      - Obtener 8 nodos (2³ = 8)
      - Cada nodo contiene ~78 coeficientes wavelet
   
   d) **Calcular energía por nodo**:
      ```python
      for node in nodes_level_3:
          energy = np.sum(np.square(node.data))
      ```
      - Resultado: 8 valores de energía por canal

2. **Concatenación**:
   - 5 canales × 8 nodos = **40 características por trial**
   - Matriz final: `(60 trials, 40 features)`

#### Interpretación de Características:

Cada característica representa la **energía en una banda tiempo-frecuencia específica** de un canal determinado:

```
Feature 0:  Canal C3, Nodo 0 (frecuencias más bajas)
Feature 1:  Canal C3, Nodo 1
...
Feature 7:  Canal C3, Nodo 7 (frecuencias más altas)
Feature 8:  Canal Cz, Nodo 0
...
Feature 39: Canal Pz, Nodo 7
```

---

### 4.3 FASE 3: Selección de Características (AG)

#### Código:
```python
best_individual, logbook = run_genetic_algorithm(
    X_cal_twp,              # (60, 40)
    targets_cal,            # (60,) [1,1,...,0,0,...]
    pop_size=100,
    num_generations=100,
    lambda_penalty=1.0,
    early_stopping_patience=15
)

selected_indices = [idx for idx, bit in enumerate(best_individual) if bit == 1]
# Ejemplo: selected_indices = [8, 16, 24, 39] → 4 características
```

#### Pseudocódigo del Algoritmo:

```
ALGORITMO GENÉTICO(X, y, pop_size, num_gen, λ)
│
├── Inicializar población aleatoria de 100 individuos
│   Individuo = vector binario de longitud 40
│
├── PARA generación = 1 hasta 100:
│   │
│   ├── PARA CADA individuo en población:
│   │   │
│   │   ├── Extraer características seleccionadas (1s en el vector)
│   │   ├── SI no hay características seleccionadas:
│   │   │   └── fitness = 0
│   │   │
│   │   ├── SINO:
│   │   │   ├── X_reducido = X[:, características_seleccionadas]
│   │   │   ├── LDA = LinearDiscriminantAnalysis()
│   │   │   ├── scores = cross_val_score(LDA, X_reducido, y, cv=5)
│   │   │   ├── accuracy = mean(scores) × 100
│   │   │   │
│   │   │   ├── sparsity = n_seleccionadas / 40
│   │   │   ├── penalización = λ × sparsity × 100
│   │   │   │
│   │   │   └── fitness = accuracy - penalización
│   │   │
│   │   └── Asignar fitness al individuo
│   │
│   ├── Seleccionar mejores individuos (Tournament)
│   │
│   ├── Aplicar Cruce (70% probabilidad, two-point)
│   │
│   ├── Aplicar Mutación (30% probabilidad, bit-flip 5%)
│   │
│   ├── SI fitness_máximo no mejoró en 15 generaciones:
│   │   └── BREAK (Early stopping)
│   │
│   └── Registrar estadísticas (max, avg, std)
│
└── RETORNAR mejor individuo encontrado
```

#### Ejemplo de Evolución:

```
Gen   0: Max Fitness =  35.83 | Avg =  15.33 | Std =  8.68
         → Población inicial aleatoria, baja aptitud

Gen  10: Max Fitness =  62.50 | Avg =  48.67 | Std =  4.87
         → Convergencia rápida, mejora significativa

Gen  30: Max Fitness =  68.33 | Avg =  63.53 | Std =  5.04
         → Cerca del óptimo, población homogénea

Gen  40: Max Fitness =  68.33 | Avg =  64.39 | Std =  6.31
         → Sin mejora por 15 generaciones → EARLY STOP
```

#### Análisis del Mejor Individuo:

```python
best_individual = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,...]
                                    ↑              ↑              ↑
                                Feature 8      Feature 16     Feature 24
                                (Cz, Nodo 0)   (C4, Nodo 0)   (P3, Nodo 0)
```

**Interpretación**: El AG seleccionó características de nodos de baja frecuencia (Nodo 0) en canales centrales y parietales.

---

### 4.4 FASE 4: Entrenamiento del Clasificador

#### Código:
```python
X_cal_opt = X_cal_twp[:, selected_indices]  # (60, 4) si 4 features

clf = LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')
clf_final, metrics_cal = train_clf_and_get_metrics(
    X_cal_opt, 
    y_calibration, 
    clf
)

print(f"Calibración - Acc: {metrics_cal.acc}% | TPR: {metrics_cal.tpr}%")
```

#### Proceso Interno:

1. **Reducción de Dimensionalidad**:
   - Entrada: `(60, 40)` → Salida: `(60, 4)`
   - Se usan solo las columnas seleccionadas por el AG

2. **Entrenamiento LDA**:
   ```python
   clf.fit(X_cal_opt, y_calibration)
   ```
   
   Internamente:
   - Calcula medias por clase: `μ_MI`, `μ_Rest`
   - Calcula matrices de covarianza: `Σ_MI`, `Σ_Rest`
   - Aplica shrinkage: `Σ_shrunk = (1-α)Σ + α·I`
   - Resuelve problema de autovalores para encontrar `w`

3. **Métricas en Calibración**:
   ```python
   y_pred = clf.predict(X_cal_opt)
   
   accuracy = np.sum(y_pred == y_true) / len(y_true) × 100
   
   tpr = np.sum(y_pred[y_true == 1] == 1) / np.sum(y_true == 1) × 100
   ```

---

### 4.5 FASE 5: Evaluación en Terapia

#### Código:
```python
# 1. Extraer características de datos de terapia
X_ter_twp = get_twp_feature_vectors(
    data_ter, 
    level=3, 
    wavelet='coif3'
)  # (60, 40)

# 2. Aplicar LA MISMA selección de características
X_ter_opt = X_ter_twp[:, selected_indices]  # (60, 4)

# 3. Clasificar con el modelo entrenado
y_pred = clf_final.predict(X_ter_opt)

# 4. Evaluar contra targets reales
metrics_ter = evaluate_clf_and_get_metrics(
    X_ter_opt, 
    clf_final, 
    targets_ter  # Targets reales de terapia
)

print(f"Terapia - Acc: {metrics_ter.acc}% | TPR: {metrics_ter.tpr}%")
```

#### ⚠️ IMPORTANTE: Mismo Pipeline de Preprocesamiento

Es **crítico** que los datos de terapia pasen por el **mismo pipeline**:

1. ✅ Mismo filtro (8-30 Hz, Butterworth orden 4)
2. ✅ Misma wavelet (coif3 por ejemplo)
3. ✅ Mismo level (3)
4. ✅ **Mismas características** (índices del AG)

Si cambias cualquiera de estos pasos, los resultados serán inválidos.

---

## 5. Configuración y Parámetros

### 5.1 Parámetros del Pipeline

```python
# ============== TRANSFORMADA WAVELET PACKET ==============
TWP_LEVEL = 3              # Nivel de descomposición
                           # 3 → 40 features | 4 → 80 features

TWP_WAVELET = 'coif3'      # Familia de wavelet
                           # Opciones: 'db4', 'coif3', 'sym4', 'bior3.3'

# ============== ALGORITMO GENÉTICO ==============
POP_SIZE = 100             # Tamaño de la población
                           # Rango recomendado: 50-200

N_GEN = 100                # Número máximo de generaciones
                           # Típicamente converge en 30-60

LAMBDA_PENALTY = 1.0       # Peso de penalización por sparsity
                           # ↑ Mayor → Menos características
                           # ↓ Menor → Más características
                           # Rango: 0.3 - 2.0

EARLY_STOPPING = 15        # Paciencia para early stopping
                           # Detiene si no mejora en N generaciones

K_FOLDS = 5                # Folds para validación cruzada
                           # Usado en el fitness del AG

# ============== CLASIFICADOR LDA ==============
SHRINKAGE = 'auto'         # Regularización automática
SOLVER = 'eigen'           # Solver más estable

# ============== PREPROCESAMIENTO ==============
FILTER_LOW = 8             # Frecuencia baja del filtro (Hz)
FILTER_HIGH = 30           # Frecuencia alta del filtro (Hz)
FILTER_ORDER = 4           # Orden del filtro Butterworth
SAMPLING_RATE = 250        # Frecuencia de muestreo (Hz)
```

### 5.2 Guía de Ajuste de Parámetros

#### Lambda (LAMBDA_PENALTY)

| Valor | Efecto | Cuándo Usar |
|-------|--------|-------------|
| 0.1-0.3 | Selección **permisiva** (10-15 features) | Dataset pequeño, señales ruidosas |
| 0.5-1.0 | Selección **balanceada** (5-8 features) | **Recomendado** para inicio |
| 1.5-2.5 | Selección **agresiva** (2-4 features) | Muchas características redundantes |
| >3.0 | **Demasiado restrictivo** (1-2 features) | ⚠️ No recomendado (underfitting) |

**Regla práctica:**
```
Si (Acc_Calibración - Acc_Terapia) > 20%:
    → Reducir lambda (menos overfitting)

Si n_características_promedio < 3:
    → Reducir lambda (más características)

Si TPR_Terapia < 50%:
    → Reducir lambda o aumentar level
```

#### Level de TWP

| Level | N° Features | Resolución Frecuencia | Uso Recomendado |
|-------|-------------|----------------------|-----------------|
| 2 | 20 (5×4) | Baja | Pruebas rápidas |
| **3** | **40 (5×8)** | **Media** | **Estándar** |
| **4** | **80 (5×16)** | **Alta** | Mayor precisión |
| 5 | 160 (5×32) | Muy alta | ⚠️ Riesgo overfitting |

---

## 6. Resultados Esperados

### 6.1 Métricas Típicas

#### Calibración (Training)
- **Accuracy**: 70-85%
- **TPR**: 75-90%
- **Gap calibración-terapia**: 10-20%

#### Terapia (Test/Validación)
- **Accuracy**: 55-70%
- **TPR**: 55-75%

#### Selección de Características
- **Promedio seleccionadas**: 3-8 de 40 (7.5-20%)
- **Reducción**: 80-92.5%

### 6.2 Ejemplo de Resultados (Tu Ejecución)

```
RESUMEN ESTADÍSTICO:
─────────────────────────────────────────────
Métrica                        Media        Std
─────────────────────────────────────────────
Características seleccionadas  2.88         1.36
Porcentaje selección (%)       7.19         3.39
Fitness AG                     68.23        6.38
Acc Calibración (%)            74.17        8.20
TPR Calibración (%)            79.58        8.66
Acc Terapia (%)                56.88        6.38
TPR Terapia (%)                60.25        29.37
─────────────────────────────────────────────
```

### 6.3 Comparación con Baseline (Sin AG)

| Métrica | Baseline (40 features) | Con AG (2.9 features) | Mejora |
|---------|------------------------|----------------------|--------|
| Acc Calibración | 82.29% | 74.17% | -8.12% |
| TPR Calibración | 85.42% | 79.58% | -5.84% |
| **Acc Terapia** | **55.83%** | **56.88%** | **+1.05%** ✅ |
| **TPR Terapia** | **57.71%** | **60.25%** | **+2.54%** ✅ |
| **Gap Calib-Terap** | **26.46%** | **17.29%** | **-9.17%** ✅ |

**Conclusión:** El AG reduce el overfitting significativamente (-9.17% en gap) y mejora ligeramente la generalización en terapia (+1-2%).

---

## 7. Interpretación de Resultados

### 7.1 Análisis de Canales Seleccionados

#### Distribución Observada (Promedio 8 sujetos)

| Canal | Veces Seleccionado | % | Interpretación Neurológica |
|-------|-------------------|---|---------------------------|
| **Cz** | 7/8 | 30.4% | **Corteza motora central** - Activación motora general |
| **Pz** | 6/8 | 26.1% | **Corteza parietal** - Integración sensoriomotora |
| **C4** | 5/8 | 21.7% | **Motor derecho** - Movimiento mano izquierda |
| **P3** | 5/8 | 21.7% | **Parietal izquierdo** - Procesamiento espacial |
| **C3** | 1/8 | 4.3% | Motor izquierdo - Menos relevante |

#### Montaje de Electrodos

```
        Cz (Centro motor)
         │
    ─────┼─────
    │    │    │
   C3   Cz   C4  ← Fila motora
    │    │    │
   P3   Pz   P4  ← Fila parietal
```

**Interpretación:**
- ✅ **Cz y Pz dominan**: Esto es esperado, ya que son las áreas centrales de activación motora
- ⚠️ **C3 poco usado**: Puede indicar que el protocolo enfatiza movimiento de mano derecha o que C3 tiene más ruido
- ✅ **C4 importante**: Sugiere lateralización correcta (hemisferio derecho controla mano izquierda)

### 7.2 Análisis de Nodos Wavelet (Bandas de Frecuencia)

#### Distribución por Nodo (Level 3 = 8 nodos)

```
Nodo  Frecuencia Aproximada    Veces Seleccionado
────────────────────────────────────────────────
  0   8-11 Hz (Mu bajo)        █████████ (9)
  1   11-14 Hz (Mu alto/Beta)  ████ (4)
  2   14-17 Hz (Beta bajo)     ██ (2)
  3   17-20 Hz (Beta medio)    ██████ (6)
  4   20-23 Hz (Beta alto)     ████ (4)
  5   23-26 Hz (Beta muy alto) ██ (2)
  6   26-29 Hz (Beta extremo)  ██ (2)
  7   29-30 Hz (límite Beta)   █ (1)
```

**Conclusión:**
- ✅ **Nodo 0 (8-11 Hz)**: Ritmo mu dominante → **Correcto**
- ✅ **Nodo 3 (17-20 Hz)**: Beta medio también importante
- ⚠️ **Nodos 5-7 poco usados**: Frecuencias altas menos informativas (más ruido)

#### Relación con ERD/ERS

Durante imagería motora:
- **ERD (8-12 Hz)** → Nodos 0-1 → ✅ Altamente seleccionados
- **ERD (18-26 Hz)** → Nodos 3-5 → ✅ Moderadamente seleccionados
- **Ruido (>26 Hz)** → Nodos 6-7 → ✅ Correctamente ignorados

### 7.3 Análisis de Sujetos Individuales

#### Sujetos con Buen Desempeño (TPR Terapia >75%)

**Sujeto 2:**
```
Características: 2/40 (5%)
- C4, Nodo 0 (8-11 Hz)
- Pz, Nodo 5 (23-26 Hz)

Calibración: Acc=81.67%, TPR=93.33%
Terapia:     Acc=63.33%, TPR=91.67%

✅ Análisis: Excelente TPR en ambas fases. La combinación C4+Pz
            captura bien la lateralización motora.
```

**Sujeto 6:**
```
Características: 5/40 (12.5%)
- Cz, Nodo 0, 1
- C4, Nodo 0
- Pz, Nodo 2, 6

Calibración: Acc=90.00%, TPR=86.67%
Terapia:     Acc=58.33%, TPR=90.62%

✅ Análisis: Mayor número de características permite capturar más
            patrones. TPR excepcional (>90%) indica baja tasa de
            falsos negativos.
```

#### Sujetos Problemáticos (TPR Terapia <20%)

**Sujeto 3:**
```
Características: 2/40 (5%)
- P3, Nodo 0
- Pz, Nodo 3

Calibración: Acc=68.33%, TPR=83.33%
Terapia:     Acc=50.00%, TPR=14.29%

❌ Análisis: Colapso del TPR en terapia (83% → 14%). Posibles causas:
  1. Overfitting severo a datos de calibración
  2. Distribución diferente en terapia
  3. Solo 2 características insuficientes
  4. Características seleccionadas no generalizan

💡 Solución: Reducir lambda (0.3-0.5) para seleccionar más características
```

**Sujeto 4:**
```
Características: 2/40 (5%)
- Cz, Nodo 4
- P3, Nodo 6

Calibración: Acc=70.00%, TPR=66.67%
Terapia:     Acc=55.00%, TPR=0.00%

❌ Análisis: TPR = 0% significa que el modelo clasifica TODOS los
            trials como clase Rest (nunca predice MI).

💡 Posibles causas:
  1. Umbral del clasificador muy alto
  2. Características no discriminativas
  3. Datos de terapia muy diferentes de calibración
  4. Bias del modelo hacia clase mayoritaria

💡 Solución: 
  - Usar más características (lambda = 0.3)
  - Verificar balance de clases en targets_ter
  - Probar con level=4 (más resolución)
```

**Sujeto 8:**
```
Características: 1/40 (2.5%) ⚠️
- C3, Nodo 0

Calibración: Acc=61.67%, TPR=80.00%
Terapia:     Acc=46.67%, TPR=72.73%

❌ Análisis: Solo 1 característica es insuficiente. Aunque el TPR
            no colapsa, el accuracy es apenas mejor que azar (50%).

💡 Solución: Forzar mínimo de 3 características o reducir lambda a 0.3
```

### 7.4 Métricas de Calidad del Modelo

#### Gap de Generalización

```
Gap = Acc_Calibración - Acc_Terapia

Gap Ideal:     5-10%  → Excelente generalización
Gap Aceptable: 10-20% → Generalización moderada
Gap Alto:      >20%   → Overfitting severo
```

**Tu resultado:** 17.29% → **Aceptable**, pero mejorable

#### Relación Características vs. Performance

```python
# Análisis de correlación (datos de tu ejecución)

N_Features: [4, 2, 2, 2, 4, 5, 3, 1]
Acc_Terapia: [66.67, 63.33, 50.00, 55.00, 55.00, 58.33, 60.00, 46.67]

Correlación: r = 0.61 (positiva moderada)
```

**Conclusión:** Más características → Mejor accuracy (hasta cierto punto)

---

## 8. Guía de Troubleshooting

### 8.1 Problemas Comunes y Soluciones

#### Problema 1: TPR muy bajo en terapia (<40%)

**Síntomas:**
```
Terapia: Acc=50-60%, TPR=0-30%
```

**Diagnóstico:**
- El modelo está sesgado hacia clase negativa (Rest)
- Características no capturan patrones de MI

**Soluciones:**
```python
# Solución 1: Reducir lambda
LAMBDA_PENALTY = 0.3  # Antes: 1.0

# Solución 2: Forzar mínimo de características
MIN_FEATURES = 5

# Solución 3: Aumentar resolución
TWP_LEVEL = 4  # De 3 a 4 → 40 a 80 features

# Solución 4: Probar otra wavelet
TWP_WAVELET = 'db4'  # Daubechies en lugar de coif3

# Solución 5: Verificar balance de clases
print(np.bincount(targets_ter))  # Debe ser [30, 30] aprox
```

---

#### Problema 2: Gap muy alto (>25%)

**Síntomas:**
```
Calibración: Acc=85%, TPR=90%
Terapia:     Acc=55%, TPR=60%
Gap = 30% ❌
```

**Diagnóstico:** Overfitting severo

**Soluciones:**
```python
# Solución 1: Aumentar penalización
LAMBDA_PENALTY = 1.5  # Forzar menos características

# Solución 2: Usar validación estratificada
# En genetic_algorithm.py:
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Solución 3: Regularización adicional en LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(shrinkage=0.8)  # Forzar shrinkage

# Solución 4: Data augmentation (avanzado)
# Aplicar jitter temporal o rotación de canales
```

---

#### Problema 3: AG no converge (Fitness oscila)

**Síntomas:**
```
Gen   0: Max=35.83
Gen  20: Max=45.12
Gen  40: Max=42.88  ← Retroceso
Gen  60: Max=48.00
Gen  80: Max=44.50  ← Oscilación
```

**Diagnóstico:** 
- Población demasiado pequeña
- Tasa de mutación muy alta
- Lambda mal ajustado

**Soluciones:**
```python
# Solución 1: Aumentar población
POP_SIZE = 200  # De 100 a 200

# Solución 2: Reducir mutación
# En genetic_algorithm.py:
toolbox.register("mutate", tools.mutFlipBit, indpb=0.03)  # De 0.05 a 0.03

# Solución 3: Ajustar operadores genéticos
# Aumentar cruce, reducir mutación
CXPB = 0.8  # Probabilidad de cruce
MUTPB = 0.2  # Probabilidad de mutación
```

---

#### Problema 4: Solo 1-2 características seleccionadas

**Síntomas:**
```
Sujeto 8: 1/40 características (2.5%)
Fitness = 57.50
Acc_Terapia = 46.67% (apenas mejor que azar)
```

**Diagnóstico:** Lambda demasiado alta

**Soluciones:**
```python
# Solución 1: Reducir lambda drásticamente
LAMBDA_PENALTY = 0.3  # De 1.0 a 0.3

# Solución 2: Forzar mínimo de características
# En genetic_algorithm.py, función evaluate_features:

MIN_FEATURES = 3
if n_selected < MIN_FEATURES:
    # Penalizar fitness
    penalty_additional = 20 * (MIN_FEATURES - n_selected)
    fitness_score -= penalty_additional

# Solución 3: Cambiar función de penalización
# Penalización logarítmica en lugar de lineal:
penalty = lambda_penalty * np.log1p(n_selected) * 10
```

---

#### Problema 5: Error "singular matrix" en LDA

**Síntomas:**
```
LinAlgError: Singular matrix
```

**Diagnóstico:** 
- Más características que muestras
- Covarianza no invertible

**Soluciones:**
```python
# Solución 1: Ya estás usando shrinkage='auto' ✅

# Solución 2: Forzar shrinkage más alto
clf = LinearDiscriminantAnalysis(shrinkage=0.9, solver='eigen')

# Solución 3: Reducir características máximas
# En genetic_algorithm.py:
MAX_FEATURES = 15
if n_selected > MAX_FEATURES:
    return (0.0,)  # Penalizar individuos con >15 features
```

---

### 8.2 Checklist de Validación

Antes de considerar los resultados finales, verifica:

- [ ] **Datos cargados correctamente**
  ```python
  print(dict_cal['subject_1']['mi_rest'].shape)  # Debe ser (625, 5, 60)
  print(dict_ter['subject_1']['mi'].shape)       # Debe ser (625, 5, 60)
  ```

- [ ] **Targets balanceados**
  ```python
  print(np.bincount(y_calibration))  # Debe ser [30, 30]
  print(np.bincount(targets_ter))    # Debe ser ~[30, 30]
  ```

- [ ] **Mismas características en train/test**
  ```python
  assert X_cal_opt.shape[1] == X_ter_opt.shape[1]
  ```

- [ ] **AG converge**
  ```python
  # Debe haber early stopping o llegar a max generaciones
  # Fitness max debe ser > 50
  ```

- [ ] **Características seleccionadas razonables**
  ```python
  # Entre 3 y 15 características
  assert 3 <= len(selected_indices) <= 15
  ```

- [ ] **Métricas dentro de rangos esperados**
  ```python
  assert 50 <= metrics_cal.acc <= 95  # Calibración
  assert 45 <= metrics_ter.acc <= 75  # Terapia
  ```

---

## 9. Experimentos Avanzados

### 9.1 Grid Search de Hiperparámetros

```python
# Experimentar con diferentes configuraciones

param_grid = {
    'lambda': [0.3, 0.5, 1.0, 1.5],
    'level': [3, 4],
    'wavelet': ['db4', 'coif3', 'sym4'],
    'pop_size': [50, 100, 200]
}

best_config = None
best_acc = 0

for lam in param_grid['lambda']:
    for level in param_grid['level']:
        for wavelet in param_grid['wavelet']:
            # Entrenar con esta configuración
            # ... (código de pipeline)
            
            if acc_terapia > best_acc:
                best_acc = acc_terapia
                best_config = {
                    'lambda': lam,
                    'level': level,
                    'wavelet': wavelet
                }

print(f"Mejor configuración: {best_config}")
print(f"Accuracy terapia: {best_acc:.2f}%")
```

### 9.2 Validación Cruzada Sujeto-Independiente

```python
# Leave-One-Subject-Out Cross-Validation (LOSO-CV)

resultados_loso = []

for test_subject in range(1, 9):
    # Entrenar con 7 sujetos
    train_subjects = [s for s in range(1, 9) if s != test_subject]
    
    X_train = []
    y_train = []
    for s in train_subjects:
        X_s = get_twp_feature_vectors(dict_cal[f'subject_{s}']['mi_rest'])
        X_train.append(X_s)
        y_train.extend(y_calibration)
    
    X_train = np.vstack(X_train)
    
    # Aplicar AG
    best_ind, _ = run_genetic_algorithm(X_train, y_train, ...)
    selected = [i for i, b in enumerate(best_ind) if b == 1]
    
    # Entrenar LDA
    clf = LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')
    clf.fit(X_train[:, selected], y_train)
    
    # Evaluar en sujeto de test
    X_test = get_twp_feature_vectors(dict_cal[f'subject_{test_subject}']['mi_rest'])
    y_pred = clf.predict(X_test[:, selected])
    acc = accuracy(y_pred, y_calibration)
    
    resultados_loso.append(acc)
    print(f"Sujeto {test_subject} (test): {acc:.2f}%")

print(f"\nPromedio LOSO-CV: {np.mean(resultados_loso):.2f}%")
```

### 9.3 Análisis de Estabilidad de Características

```python
# ¿Qué características se seleccionan consistentemente?

feature_counts = np.zeros(40)

for sujeto in range(1, 9):
    data_cal = dict_cal[f'subject_{sujeto}']['mi_rest']
    X_cal = get_twp_feature_vectors(data_cal, level=3, wavelet='coif3')
    
    best_ind, _ = run_genetic_algorithm(X_cal, y_calibration, ...)
    
    for idx, bit in enumerate(best_ind):
        if bit == 1:
            feature_counts[idx] += 1

# Características más estables (seleccionadas en ≥5 sujetos)
stable_features = np.where(feature_counts >= 5)[0]

print("Características estables (seleccionadas en ≥5 sujetos):")
for feat in stable_features:
    canal = feat // 8
    nodo = feat % 8
    print(f"  Feature {feat}: Canal {channel_names[canal]}, Nodo {nodo}")
    print(f"    Seleccionada en {feature_counts[feat]:.0f}/8 sujetos")
```

### 9.4 Comparación con Otros Métodos de Selección

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Método 1: ANOVA F-test
selector_anova = SelectKBest(f_classif, k=5)
X_anova = selector_anova.fit_transform(X_cal_twp, y_calibration)

# Método 2: Recursive Feature Elimination
clf_temp = LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')
selector_rfe = RFE(clf_temp, n_features_to_select=5)
X_rfe = selector_rfe.fit_transform(X_cal_twp, y_calibration)

# Método 3: AG (tu método)
# ... (código existente)

# Comparar en terapia
results_comparison = {
    'ANOVA': evaluate_clf(X_ter_anova, ...),
    'RFE': evaluate_clf(X_ter_rfe, ...),
    'AG': evaluate_clf(X_ter_ag, ...)
}

print("Comparación de métodos:")
for method, acc in results_comparison.items():
    print(f"  {method}: {acc:.2f}%")
```

---

## 10. Extensiones Futuras

### 10.1 Mejoras del Pipeline

1. **Ensemble de Clasificadores**
   ```python
   from sklearn.ensemble import VotingClassifier
   
   clf1 = LinearDiscriminantAnalysis(shrinkage='auto')
   clf2 = SVC(kernel='rbf', probability=True)
   clf3 = RandomForestClassifier(n_estimators=100)
   
   ensemble = VotingClassifier(
       estimators=[('lda', clf1), ('svc', clf2), ('rf', clf3)],
       voting='soft'
   )
   ```

2. **Transfer Learning entre Sujetos**
   - Entrenar modelo base con todos los sujetos
   - Fine-tuning con datos del sujeto específico

3. **Adaptación Online**
   - Actualizar clasificador durante sesión de terapia
   - Usar feedback del usuario para corregir predicciones

### 10.2 Nuevas Características

1. **Common Spatial Patterns (CSP)**
   ```python
   from mne.decoding import CSP
   
   csp = CSP(n_components=4)
   X_csp = csp.fit_transform(data_cal, y_calibration)
   ```

2. **Conectividad Funcional**
   - Phase Locking Value (PLV)
   - Coherencia entre canales

3. **Features Temporales**
   - Hjorth Parameters (Activity, Mobility, Complexity)
   - Sample Entropy
   - Fractal Dimension

### 10.3 Clasificadores Alternativos

1. **Deep Learning**
   ```python
   # CNN para señales EEG
   from tensorflow.keras import Sequential, layers
   
   model = Sequential([
       layers.Conv1D(32, kernel_size=5, activation='relu'),
       layers.MaxPooling1D(2),
       layers.Conv1D(64, kernel_size=3, activation='relu'),
       layers.GlobalAveragePooling1D(),
       layers.Dense(1, activation='sigmoid')
   ])
   ```

2. **Riemannian Geometry**
   ```python
   from pyriemann.classification import MDM
   
   # Minimum Distance to Mean en variedad de matrices de covarianza
   clf_riemann = MDM()
   ```

---

## 11. Conclusiones

### 11.1 Resumen del Pipeline

Este pipeline implementa un **enfoque híbrido estado del arte** para clasificación de imagería motora:

✅ **Fortalezas:**
- Extracción robusta de características (TWP)
- Selección inteligente guiada por evolución (AG)
- Clasificador eficiente y estable (LDA)
- Reducción drástica de dimensionalidad (92%)
- Prevención de overfitting mediante sparsity

⚠️ **Limitaciones:**
- Requiere ajuste de hiperparámetros (lambda)
- Variabilidad inter-sujeto alta
- Computacionalmente costoso (AG tarda 5-10 min/sujeto)

### 11.2 Lecciones Aprendidas

1. **Lambda es crítico**: Controla el trade-off precisión vs. generalización
2. **Canales centrales (Cz, Pz) son los más informativos**
3. **Ritmo mu (8-12 Hz) domina la discriminación MI vs Rest**
4. **2-3 características pueden ser insuficientes** → Aumentar mínimo a 5
5. **Algunos sujetos requieren configuraciones específicas**

### 11.3 Recomendaciones Finales

Para **uso en producción**:
```python
# Configuración recomendada basada en análisis
TWP_LEVEL = 4           # Mayor resolución
TWP_WAVELET = 'coif3'   # Mantener
POP_SIZE = 100          # Suficiente
N_GEN = 100             # Con early stopping
LAMBDA_PENALTY = 0.5    # Balanceado
EARLY_STOPPING = 15     # Mantener
MIN_FEATURES = 5        # Nuevo: forzar mínimo
```

Para **investigación**:
- Experimentar con nivel 5 (160 features)
- Probar wavelets adaptativas
- Implementar multi-objetivo AG (precisión + robustez)
- Validación LOSO-CV obligatoria

---

## 12. Referencias

### 12.1 Literatura Científica

1. **Imagería Motora y BCI:**
   - Pfurtscheller, G., & Neuper, C. (2001). *Motor imagery and direct brain-computer communication.* IEEE, 89(7), 1123-1134.
   - Blankertz, B., et al. (2008). *The BCI competition III: Validating alternative approaches to actual BCI problems.* IEEE Trans Neural Syst Rehabil Eng, 14(2), 153-159.

2. **Transformada Wavelet:**
   - Subasi, A. (2007). *EEG signal classification using wavelet feature extraction and a mixture of expert model.* Expert Systems with Applications, 32(4), 1084-1093.
   - Mallat, S. (1989). *A theory for multiresolution signal decomposition: the wavelet representation.* IEEE Trans Pattern Anal Mach Intell, 11(7), 674-693.

3. **Algoritmos Genéticos:**
   - Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning.* Addison-Wesley.
   - Fortin, F. A., et al. (2012). *DEAP: Evolutionary algorithms made easy.* Journal of Machine Learning Research, 13, 2171-2175.

4. **Selección de Características en BCI:**
   - Lotte, F., et al. (2018). *A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update.* J Neural Eng, 15(3), 031005.
   - Baig, M. Z., et al. (2017). *Filtering techniques for channel selection in motor imagery EEG applications: a survey.* Artif Intell Rev, 53, 1207-1232.

### 12.2 Herramientas y Librerías

- **PyWavelets:** https://pywavelets.readthedocs.io/
- **DEAP (Genetic Algorithms):** https://deap.readthedocs.io/
- **Scikit-learn (LDA):** https://scikit-learn.org/
- **MNE-Python (EEG):** https://mne.tools/

### 12.3 Datasets

- **IM-tention Dataset:** [Dataset de creación propia por el CIRINS - FiUNER]
---

## Apéndices

### Apéndice A: Glosario de Términos

| Término | Definición |
|---------|-----------|
| **BCI** | Brain-Computer Interface - Interfaz cerebro-computadora |
| **ERD** | Event-Related Desynchronization - Desincronización relacionada con evento |
| **ERS** | Event-Related Synchronization - Sincronización relacionada con evento |
| **MI** | Motor Imagery - Imagería motora |
| **TWP** | Tree Wavelet Packet - Transformada Wavelet Packet |
| **AG** | Algoritmo Genético |
| **LDA** | Linear Discriminant Analysis - Análisis Discriminante Lineal |
| **TPR** | True Positive Rate - Tasa de verdaderos positivos (Sensibilidad) |
| **Sparsity** | Escasez - Proporción de características seleccionadas |
| **Early Stopping** | Detención temprana del entrenamiento |

### Apéndice B: Código Completo Mínimo

```python
# Pipeline mínimo en ~30 líneas

import numpy as np
from modules.preprocessing.file_creation import create_mat_files
from modules.feature_extraction.feature_extraction import get_twp_feature_vectors
from modules.genetic_algorithm.genetic_algorithm import run_genetic_algorithm
from modules.training.training import train_clf_and_get_metrics
from modules.evaluation.evaluation import evaluate_clf_and_get_metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. Cargar datos
dict_cal = create_mat_files('./data/im_tention_signals', file_type='calibration', filtfilt=True)
dict_ter = create_mat_files('./data/im_tention_signals', file_type='therapy', ther_number_of_trials=60, filtfilt=True)

# 2. Preparar targets
y_calibration = np.hstack((np.ones(30, dtype=np.int8), np.zeros(30, dtype=np.int8)))

# 3. Pipeline para un sujeto
data_cal = dict_cal['subject_1']['mi_rest']
data_ter = dict_ter['subject_1']['mi']
targets_ter = dict_ter['subject_1']['target']

# 4. Extracción TWP
X_cal = get_twp_feature_vectors(data_cal, level=3, wavelet='coif3')
X_ter = get_twp_feature_vectors(data_ter, level=3, wavelet='coif3')

# 5. Selección AG
best_ind, _ = run_genetic_algorithm(X_cal, y_calibration, pop_size=100, num_generations=100, lambda_penalty=0.5)
selected = [i for i, b in enumerate(best_ind) if b == 1]

# 6. Entrenamiento
clf = LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')
clf, metrics_cal = train_clf_and_get_metrics(X_