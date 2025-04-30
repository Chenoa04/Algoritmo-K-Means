# Algoritmo-K-Means
Este repositorio implementa tres métodos principales para determinar el número óptimo de clústeres en el algoritmo K-Means.
1. Método del Codo (Elbow Method)
   - Analiza la suma de cuadrados intra-cluster (WCSS)
   - Busca el punto donde la disminución en WCSS comienza a nivelarse
   - Ventajas: Simple, intuitivo visualmente
   - Desventajas: Subjetivo (a veces no hay un codo claro)

2. Método Silhouette
   - Calcula el coeficiente de silueta para cada punto
   - Evalúa qué tan bien está cada punto en su clúster comparado con otros clústeres
   - El k óptimo maximiza el puntaje promedio de silueta
   - Ventajas: Considera tanto cohesión como separación
   - Desventajas: Computacionalmente más costoso

3. Gap Statistic
   - Compara la variación intra-cluster con su valor esperado bajo una distribución de referencia nula
   - El k óptimo es el menor valor donde el gap statistic es máximo
   - Ventajas: Comparación con distribución nula, más objetivo
   - Desventajas: Más complejo, requiere más cómputo

Cada método generará:
1. Gráfico con la métrica correspondiente
2. Valor sugerido de k (número óptimo de clústeres)
3. Visualización del clustering final
