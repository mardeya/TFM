# Modelització d’epidèmies SIR en xarxes: percolació i enfocaments basats en arestes

Repositori del Treball Final de Màster -- Màster en Sistemes Intel·ligents.

## Estructura del repositori

- La carpeta __funcions__ conté el conjunt de funcions generals necessàries per dur a terme les simulacions i la modelització emprada per generar les figures.

- La carpeta __figures_memoria__ conté les imatges generades per a la memòria.

- Finalment, la carpeta __estudi_xarxes_reals__ conté un estudi realitzat sobre dues xarxes reals obtingudes de l’_Stanford Network Analysis Project (SNAP)_:
  https://snap.stanford.edu/data/index.html.
  - Xarxa 1. _Social circles: Facebook_
    - McAuley, J., & Leskovec, J. (2012). _Learning to Discover Social Circles in Ego Networks_. Advances in Neural Information Processing Systems.
  - Xarxa 2. _Facebook Large Page--Page Network_
    - Rozemberczki, B., Allen, C., & Sarkar, R. (2019). _Multi-scale Attributed Node Embedding_. arXiv:1909.13021.

Fora d’aquestes carpetes, es troben els scripts que han generat les imatges, i en el seu nom s’indica el número de figura que ocupen dins la memòria.

## Agraïments 
Per dur a terme les simulacions i generar les figures, aquest treball s’ha utilitzat en el paquet __NetworkX__ per facilitar el treball amb xarxes:
  - Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). _Exploring network structure, dynamics, and function using NetworkX_. In _Proceedings of the 7th Python in Science Conference (SciPy2008)_, 11–15.
