## Aplicació dels models en xarxes reals

A més de l’estudi sobre xarxes aleatòries, també s’analitza el comportament dels models sobre dues xarxes reals de Facebook obtingudes de SNAP.

### Xarxa 1: Social circles: Facebook

La primera xarxa correspon al conjunt **Social circles: Facebook**, construït a partir d’ego-networks anonimitzades. Es tracta d’una xarxa amb un coeficient d’agrupament alt, de manera que les hipòtesis locals de l’EBCM estàndard deixen de ser del tot adequades.

A les figures associades es compara l’evolució temporal de la fracció infectada \(I(t)\) obtinguda amb simulacions de Gillespie amb tres prediccions deterministes: l’EBCM estàndard, l’EBCM amb correlacions de grau i una correcció empírica per tenir en compte l’efecte del coeficient d’agrupament. Les diferents subfigures mostren diversos valors de \(tau\), mantenint \(gamma = 1\), per observar com canvia l’ajust segons la intensitat de transmissió.

Els resultats mostren que l’EBCM estàndard i la variant amb correlacions de grau capturen raonablement el moment del brot, però no sempre descriuen bé la forma del pic epidèmic. En aquest cas, la correcció empírica permet retardar i eixamplar el brot, i dona una aproximació qualitativament més propera a les simulacions, sobretot per a valors petits de \(tau\).

### Xarxa 2: Facebook Large Page--Page Network

La segona xarxa correspon al conjunt **Facebook Large Page--Page Network**, on els nodes representen pàgines de Facebook i les arestes indiquen relacions entre aquestes pàgines. En comparació amb la Xarxa 1, aquí el coeficient d’agrupament és menor, però les correlacions de mescla i l’heterogeneïtat estructural tenen un paper més rellevant.

A les figures corresponents es representa novament l’evolució de \(I(t)\) per a diversos valors de \(tau\), comparant les simulacions de Gillespie amb l’EBCM estàndard i amb la seva extensió amb correlacions de grau. Igual que abans, cada subfigura permet veure com varia la qualitat de l’ajust en funció del règim epidemiològic considerat.

En aquest cas, l’EBCM estàndard ja reprodueix força bé l’ordre de magnitud i la posició del pic. Tot i així, la versió amb correlacions de grau introdueix una millora lleu al voltant del màxim, fet coherent amb la presència de patrons de mescla que no queden recollits només amb la distribució de graus.

### Interpretació general de les figures

En conjunt, les figures mostren dos mecanismes diferents de desviació respecte de les hipòtesis de l’EBCM estàndard. A la **Xarxa 1**, el factor dominant és l’alt coeficient d’agrupament, que introdueix dependències locals i redueix la validesa de l’aproximació localment arbòria. A la **Xarxa 2**, en canvi, les discrepàncies s’expliquen principalment per les correlacions de graus i per l’estructura heterogènia de la xarxa.
