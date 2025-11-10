# VAML Coursework Group 29 #

**Members involved: Gabriella Keisha Andini(40392749), Su Thinzar Thaw(40392455), Yu-Zhi Wong(40374472)**

*Notes update (40392749):*
- Under my branch 40392749-keisha i have it with hog and full image tested with all method (SVM RBF, SVM Linear, Nearest neighbour, anD K-NN)
- Run a-d, then pick which method to test 
- Training automatic system result: 

BEST ACCURACY: SVM-RBF-HOG (C=10,ks=72.9938) (99.89%)
FASTEST MODEL: SVM-Linear-HOG (C=0.01) (0.000049s)

ANALYSIS & RECOMMENDATIONS:
Best HOG model: SVM-RBF-HOG (C=10,ks=72.9938) (99.89%)
LINEAR vs RBF SVM (HOG):
Linear: 99.56% | RBF: 99.89% | Difference: -0.33%

- All data normalised first for a stable accuracy result 
- Testing the classification system result return and summarise on m_compare_all.m 
- using split 70/30
- next step: update other combination, use the other feature descriptor (HOG+PCA, HOG+LDA, etc), showing clearer evaluation, using cross validation, etc 
- second next: detection implemention and finalize for initial demo preparation 

