# Documentación del Pipeline: Clasificación de Imagería Motora con AG + TWP + LDA

**Autor:** [Tu Nombre]  
**Fecha:** Noviembre 2025  
**Versión:** 1.0

---

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
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

Este proyecto implementa un **sistema de clasificación de señales EEG** para detectar **Imagería Motora (MI - Motor Imagery)** utilizando un enfoque híbrido que combina:

- **Transformada Wavelet Packet (TWP)** para extracción de características
- **Algoritmo Genético (AG)** para selección óptima de características
- **Linear Discriminant Analysis (LDA)** como clasificador

El objetivo es distinguir entre dos estados mentales:
- **Motor Imagery (MI)**: Imaginación de movimiento
- **Rest (Reposo)**: Estado de reposo mental

### 1.2 Contexto Clínico

Este sistema es parte de una **Brain-Computer Interface (BCI)** diseñada para aplicaciones de:
- Rehabilitación motora post-ACV
- Control de prótesis robóticas
- Comunicación asistida

---

## 2. Base Teórica

### 2.1 Imagería Motora y Ritmos Sensoriomotores

#### ¿Qué es la Imagería Motora?

La **imagería motora** es la simulación mental de un movimiento sin ejecución física real. Durante este proceso, se activan áreas motoras del cerebro similares a las que se activarían durante el movimiento real.

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

#### Fundamento Matemático

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
2. ✅ Misma wavelet (coif3)
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

| Métrica | Baseline (40 features) | Con AG (2.9 features