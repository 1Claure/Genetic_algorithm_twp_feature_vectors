import numpy as np
from scipy.signal import welch
import pywt

factor_escala = 22369.62  # n_muestras/mV

def get_twp_feature_vectors(data_per_subject, wavelet='coif3', level=3, 
                           normalize=False, log_transform=False):
    """
    Extraer vectores de características usando Energía de Transformada Wavelet Packet (TWP).
    
    MEJORAS:
    - Parámetros por defecto actualizados (wavelet='coif3', level=3)
    - Procesa TODOS los canales individualmente
    - Opción de normalización y transformación logarítmica
    - Mejor documentación
    
    Args:
        data_per_subject (array): Matriz con forma (n_samples, n_channels, n_trials)
        wavelet (str): Tipo de wavelet (default: 'coif3')
                      Opciones comunes: 'db4', 'coif3', 'sym4', 'bior3.3'
        level (int): Nivel de descomposición (default: 3)
                    level=3 → 2^3=8 nodos/canal → 5 canales × 8 = 40 características
                    level=4 → 2^4=16 nodos/canal → 5 canales × 16 = 80 características
        normalize (bool): Si True, normaliza la energía por la energía total del trial
        log_transform (bool): Si True, aplica log(1+energy) para estabilizar varianza
    
    Returns:
        array: Matriz de características con forma (n_trials, n_features_twp)
              donde n_features_twp = n_channels × 2^level
    
    Ejemplo:
        >>> data = np.random.randn(500, 5, 60)  # 500 samples, 5 channels, 60 trials
        >>> features = get_twp_feature_vectors(data, level=3)
        >>> print(features.shape)  # (60, 40) = 60 trials × (5 channels × 8 nodes)
    """
    
    n_samples, n_channels, n_trials = data_per_subject.shape
    twp_feature_list = []

    for i in range(n_trials):
        trial_features = []
        
        # Iterar sobre cada canal
        for j in range(n_channels):
            # Extraer señal de un único canal para este trial
            signal_channel = data_per_subject[:, j, i]
            
            # Descomposición Wavelet Packet (TWP)
            wp = pywt.WaveletPacket(data=signal_channel, 
                                   wavelet=wavelet, 
                                   mode='symmetric', 
                                   maxlevel=level)
            
            # Extraer energía de todos los nodos del nivel más profundo
            nodes_at_level = wp.get_level(level)
            
            channel_energies = []
            for node in nodes_at_level:
                # Calcular energía (suma de cuadrados)
                energy = np.sum(np.square(node.data))
                channel_energies.append(energy)
            
            # Normalización opcional (por canal)
            if normalize:
                total_energy = np.sum(channel_energies) + 1e-10  # Evitar división por cero
                channel_energies = [e / total_energy for e in channel_energies]
            
            # Transformación logarítmica opcional
            if log_transform:
                channel_energies = [np.log1p(e) for e in channel_energies]  # log(1+x)
            
            trial_features.extend(channel_energies)
        
        twp_feature_list.append(trial_features)
    
    return np.array(twp_feature_list)

# ============================================================================
# FUNCIÓN AUXILIAR: VISUALIZAR IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================================
def analyze_selected_features(selected_indices, n_channels=5, level=3, 
                              channel_names=None):
    """
    Analizar qué canales y bandas de frecuencia fueron seleccionados por el AG.
    
    Args:
        selected_indices: Lista de índices de características seleccionadas
        n_channels: Número de canales (default: 5)
        level: Nivel de descomposición TWP (default: 3)
        channel_names: Lista de nombres de canales (opcional)
    
    Returns:
        dict: Diccionario con conteo por canal y nodo
    """
    
    if channel_names is None:
        channel_names = [f"Canal {i+1}" for i in range(n_channels)]
    
    n_nodes = 2 ** level
    
    # Conteo por canal
    channel_counts = {name: 0 for name in channel_names}
    node_counts = {i: 0 for i in range(n_nodes)}
    
    for idx in selected_indices:
        canal = idx // n_nodes
        nodo = idx % n_nodes
        
        channel_counts[channel_names[canal]] += 1
        node_counts[nodo] += 1
    
    print(f"\n{'='*60}")
    print(f"  ANÁLISIS DE CARACTERÍSTICAS SELECCIONADAS")
    print(f"{'='*60}")
    print(f"Total seleccionadas: {len(selected_indices)} de {n_channels * n_nodes}")
    print(f"\n📊 Distribución por Canal:")
    for name, count in channel_counts.items():
        bar = '█' * (count * 2)
        print(f"  {name:12s}: {count:2d} {bar}")
    
    print(f"\n🎵 Distribución por Nodo Wavelet:")
    for nodo, count in node_counts.items():
        bar = '█' * (count * 2)
        print(f"  Nodo {nodo:2d}: {count:2d} {bar}")
    print(f"{'='*60}\n")
    
    return {'channel_counts': channel_counts, 'node_counts': node_counts}

# ============================================================================
# FUNCIÓN AUXILIAR: COMPARAR WAVELETS
# ============================================================================
def compare_wavelets(data_sample, wavelets=['db4', 'coif3', 'sym4', 'bior3.3'], 
                    level=3):
    """
    Comparar diferentes familias de wavelets en los datos.
    
    Args:
        data_sample: Muestra de datos (n_samples, n_channels, n_trials)
        wavelets: Lista de wavelets a comparar
        level: Nivel de descomposición
    
    Returns:
        dict: Resultados de cada wavelet
    """
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"  COMPARACIÓN DE WAVELETS")
    print(f"{'='*60}")
    
    for wavelet in wavelets:
        try:
            features = get_twp_feature_vectors(data_sample, 
                                              wavelet=wavelet, 
                                              level=level)
            
            # Calcular estadísticas básicas
            mean_energy = np.mean(features)
            std_energy = np.std(features)
            
            results[wavelet] = {
                'features': features,
                'mean': mean_energy,
                'std': std_energy
            }
            
            print(f"\n{wavelet:10s} | Media: {mean_energy:10.2f} | Std: {std_energy:10.2f}")
            
        except Exception as e:
            print(f"{wavelet:10s} | ERROR: {e}")
    
    print(f"{'='*60}\n")
    
    return results