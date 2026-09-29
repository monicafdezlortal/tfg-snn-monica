# \# TFG — Spiking Neural Networks en plataformas neuromórficas (prototipo)

# 

# \## Objetivo

# Prototipar una red neuronal de spiking (SNN) en software y preparar su paso a hardware (VHDL/FPGA), evaluando rendimiento y eficiencia.

# 

# \## Caso de uso (baseline)

# \- Dataset: MNIST (o CIFAR-10)

# \- Modelo: SNN simple (baseline)

# \- Notebook: ver carpeta /notebooks

# 

# \## Métricas que voy a reportar

# \### Software

# \- Accuracy (test)

# \- Timesteps (T)

# \- (si es posible) SynOps / sparsidad de spikes

# \- (si es posible) estimación de energía proxy

# 

# \### Hardware (FPGA/VHDL)

# \- Recursos: LUT, FF, BRAM, DSP

# \- Fmax

# \- Latencia por timestep (o ciclos/step)

# \- Potencia estimada

# \- Métricas por neurona (escalabilidad)

# 

# \## Estructura del repositorio

# \- notebooks/: experimentos en Colab

# \- src/: código Python limpio

# \- hw/: VHDL y testbenches

# \- results/: CSV con métricas y figuras

# \- docs/: memoria y documentación

