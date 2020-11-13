#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define MAX_CHAINE 100

#define CALLOC(ptr, nr, type)   if (!(ptr = (type *) calloc((size_t)(nr), sizeof(type)))) { \
                                    printf("Erreur lors de l'allocation memoire \n") ;      \
                                    exit (-1);                                              \
                                }


#define FOPEN(fich,fichier,sens)    if ((fich=fopen(fichier,sens)) == NULL) {                   \
                                        printf("Probleme d'ouverture du fichier %s\n",fichier); \
                                        exit(-1);                                               \
                                    }

#define MIN(a, b)         (a < b ? a : b)
#define MAX(a, b)         (a > b ? a : b)

#define MAX_VALEUR         255
#define MIN_VALEUR         0

#define NBPOINTSPARLIGNES 15

#define false 0
#define true 1
#define boolean int

// Clock
#define initTimer struct timeval tv1, tv2; struct timezone tz
#define startTimer gettimeofday(&tv1, &tz)
#define stopTimer gettimeofday(&tv2, &tz)
#define tpsCalcul ((tv2.tv_sec-tv1.tv_sec)*1000000L + (tv2.tv_usec-tv1.tv_usec)) / 1000


/* KERNEL CUDA */
__global__ void add_vec_scalaire_gpu(int **image, int **res, long N, int x, int le_min, float etalement) {
    long i = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    int j;
    if (i < N) {
        for (j = 0 ; j < x ; j++) {
            res[i][j] = ((image[i][j] - le_min) * etalement);
        }
    }
}


int main(argc, argv) int argc; char* argv[]; {
    /*========================================================================*/
    /* Declaration de variables et allocation memoire */
    /*========================================================================*/

    int i, j, n;

    int LE_MIN = MAX_VALEUR;
    int LE_MAX = MIN_VALEUR;

    float ETALEMENT = 0.0;

    int** image;
    int** resultat;
    int X, Y, x, y;
    int TailleImage;

    int NbResultats, quelle_ligne, lignes;
    int* la_ligne;

    int P;

    FILE* Src, * Dst;

    char SrcFile[MAX_CHAINE];
    char DstFile[MAX_CHAINE];

    char ligne[MAX_CHAINE];

    boolean inverse = false;

    char *Chemin;

    initTimer; //

    /*========================================================================*/
    /* Recuperation des parametres                                                */
    /*========================================================================*/

    sscanf(argv[1], "%s", SrcFile);

    sprintf(DstFile, "%s.new", SrcFile);

    /*========================================================================*/
    /* Recuperation de l'endroit ou l'on travail                                */
    /*========================================================================*/

    CALLOC(Chemin, MAX_CHAINE, char);
    Chemin = getenv("PWD");
    printf("Repertoire de travail : %s \n\n", Chemin);

    /*========================================================================*/
    /* Ouverture des fichiers                                                */
    /*========================================================================*/

    printf("Operations sur les fichiers\n");

    FOPEN(Src, SrcFile, "r");
    printf("\t Fichier source ouvert (%s) \n", SrcFile);

    FOPEN(Dst, DstFile, "w");
    printf("\t Fichier destination ouvert (%s) \n", DstFile);

    /*========================================================================*/
    /* On effectue la lecture du fichier source */
    /*========================================================================*/

    printf("\t Lecture entete du fichier source ");

    for (i = 0; i < 2; i++) {
        fgets(ligne, MAX_CHAINE, Src);
        fprintf(Dst, "%s", ligne);
    }

    fscanf(Src, " %d %d\n", &X, &Y);
    fprintf(Dst, " %d %d\n", X, Y);

    fgets(ligne, MAX_CHAINE, Src);        /* Lecture du 255         */
    fprintf(Dst, "%s", ligne);

    printf(": OK \n");

    /*========================================================================*/
    /* Allocation memoire pour l'image source et l'image resultat                 */
    /*========================================================================*/

    CALLOC(image, Y + 1, int*);
    CALLOC(resultat, Y + 1, int*);
    for (i = 0;i < Y;i++) {
        CALLOC(image[i], X + 1, int);
        CALLOC(resultat[i], X + 1, int);
        for (j = 0;j < X;j++) {
            image[i][j] = 0;
            resultat[i][j] = 0;
        }
    }
    printf("\t\t Initialisation de l'image [%d ; %d] : Ok \n", X, Y);

    TailleImage = X * Y;

    x = 0;
    y = 0;

    lignes = 0;

    /*========================================================================*/
    /* Lecture du fichier pour remplir l'image source                         */
    /*========================================================================*/

    while (!feof(Src)) {
        n = fscanf(Src, "%d", &P);
        image[y][x] = P;
        LE_MIN = MIN(LE_MIN, P);
        LE_MAX = MAX(LE_MAX, P);
        x++;
        if (n == EOF || (x == X && y == Y - 1)) {
            break;
        }
        if (x == X) {
            x = 0;
            y++;
        }
    }
    fclose(Src);
    printf("\t Lecture du fichier image : Ok \n\n");

    /*========================================================================*/
    /* Calcul du facteur d'etalement                                        */
    /*========================================================================*/

    if (inverse) {
        ETALEMENT = 0.2;
    }
    else {
        ETALEMENT = (float)(MAX_VALEUR - MIN_VALEUR) / (float)(LE_MAX - LE_MIN);
    }

    /*========================================================================*/
    /* Calcul de chaque nouvelle valeur de pixel                              */
    /*========================================================================*/

    int tailleVecteur = Y;
    long blocksize = 1

    int **cudaVec;
    int **cudaRes;

    // Select cuda GPU device to use (if multiple device)
    cudaSetDevice(0); // TODO make it as args ?

    if (cudaMalloc((void **)&cudaVec, TailleImage) == cudaErrorMemoryAllocation) {
        printf("Allocation memoire qui pose probleme (cudaVec) \n");
    }
    if (cudaMalloc((void **)&cudaRes, TailleImage)  == cudaErrorMemoryAllocation) {
        printf("Allocation memoire qui pose probleme (cudaRes) \n");
    }

    long dimBlock = blocksize;
    long dimGrid = tailleVecteur/blocksize;
    if ((tailleVecteur % blocksize) != 0) {
        dimGrid++;
    }

    int res = cudaMemcpy(&cudaVec[0], image[0], TailleImage * sizeof(int), cudaMemcpyHostToDevice);

    int res = cudaMemcpy(&cudaVec[0], &image[0], TailleImage * sizeof(int), cudaMemcpyHostToDevice);

    printf("Copy CPU -> GPU %d \n",res);

    startTimer;
    add_vec_scalaire_gpu<<<dimGrid, dimBlock>>>(cudaVec, cudaRes, Y, X, LE_MIN, ETALEMENT);
    stopTimer;

    cudaMemcpy(&resultat[0], &cudaRes[0], TailleImage * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(cudaVec);
    cudaFree(cudaRes);

    /*========================================================================*/
    /* Sauvegarde de l'image dans le fichier resultat                         */
    /*========================================================================*/

    n = 0;
    for (i = 0; i < Y; i++) {
        for (j = 0; j < X; j++) {

            fprintf(Dst, "%3d ", resultat[i][j]);
            n++;
            if (n == NBPOINTSPARLIGNES) {
                n = 0;
                fprintf(Dst, "\n");
            }
        }
    }

    fprintf(Dst, "\n");
    fclose(Dst);

    printf("\n");

    /*========================================================================*/
    /* Fin du programme principal        */
    /*========================================================================*/

    exit(0);
}
