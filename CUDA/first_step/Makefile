all :  VecteurCPU VecteurGPU

VecteurCPU : Vecteur.c
	gcc -o VecteurCPU Vecteur.c

VecteurGPU : Vecteur.cu
	nvcc -o VecteurGPU Vecteur.cu
clean :
	rm VecteurCPU VecteurGPU
