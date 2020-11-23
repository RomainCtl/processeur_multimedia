#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

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
#define tpsCalcul ((tv2.tv_sec-tv1.tv_sec)*1000000L + (tv2.tv_usec-tv1.tv_usec))


int main(int argc, char* argv[]) {
    /*========================================================================*/
    /* Declaration de variables et allocation memoire */
    /*========================================================================*/

    if (argc < 2) {
        printf("Usage: ./CodeSequentiel <path_to_image> [<num_threads>]\n");
        exit(0);
    }

    long num_threads = 1; // default value
    if (argc == 3) {
        num_threads = atoi(argv[2]);
    }

    int i, n;

    int LE_MIN = MAX_VALEUR;
    int LE_MAX = MIN_VALEUR;

    float ETALEMENT = 0.0;

    int* resultat;
    int* image;
    int X, Y, x, y;
    int TailleImage;

    int P;

    FILE* Src, * Dst;

    char SrcFile[MAX_CHAINE];
    char DstFile[MAX_CHAINE];

    char ligne[MAX_CHAINE];

    boolean inverse = false;

    char* Chemin;

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

    TailleImage = X * Y;

    CALLOC(image, TailleImage, int);
    CALLOC(resultat, TailleImage, int);
    for (i = 0;i < TailleImage;i++) {
        image[i] = 0;
        resultat[i] = 0;
    }

    x = 0;
    y = 0;

    printf("\t\t Initialisation de l'image [%d ; %d] : Ok \n", X, Y);

    /*========================================================================*/
    /* Lecture du fichier pour remplir l'image source                         */
    /*========================================================================*/

    while (!feof(Src)) {
        n = fscanf(Src, "%d", &P);
        image[y + x] = P;
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

    int pixel_per_thread = TailleImage / num_threads + 1;
    int th_id;

    omp_set_num_threads(num_threads);
    printf("%ld threads !\n", num_threads);

    startTimer;
#pragma omp parallel private(i, th_id)
    {
        th_id = omp_get_thread_num();
        for (i = th_id * pixel_per_thread; i < (th_id + 1) * pixel_per_thread; i++) {
            resultat[i] = ((image[i] - LE_MIN) * ETALEMENT);
        }
    }
    stopTimer;

    printf("Duration %ld", tpsCalcul);


    /*========================================================================*/
    /* Sauvegarde de l'image dans le fichier resultat                         */
    /*========================================================================*/

    n = 0;
    for (i = 0; i < TailleImage; i++) {
        fprintf(Dst, "%3d ", resultat[i]);
        n++;
        if (n == NBPOINTSPARLIGNES) {
            n = 0;
            fprintf(Dst, "\n");
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
