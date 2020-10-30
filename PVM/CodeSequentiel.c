/*==============================================================================*/
/* Programme 	: CodeSequentiel.c						*/
/* Auteur 	: Daniel CHILLET						*/
/* Date 	: Decembre 2011							*/
/* 										*/
/*==============================================================================*/


#include <stdlib.h>
#include <stdio.h>
#include "pvm3.h"

#define MAX_CHAINE 100
#define MAX_HOSTS 100
#define MAX_PARAM 10



#define CALLOC(ptr, nr, type) 		if (!(ptr = (type *) calloc((size_t)(nr), sizeof(type)))) {		\
						printf("Erreur lors de l'allocation memoire \n") ; 		\
						exit (-1);							\
					} 


#define FOPEN(fich,fichier,sens) 	if ((fich=fopen(fichier,sens)) == NULL) { 				\
						printf("Probleme d'ouverture du fichier %s\n",fichier);		\
						exit(-1);							\
					} 
				
#define MIN(a, b) 	(a < b ? a : b)
#define MAX(a, b) 	(a > b ? a : b)

#define MAX_VALEUR 	255
#define MIN_VALEUR 	0

#define NBPOINTSPARLIGNES 15

#define false 0
#define true 1
#define boolean int

#define MAITRE_ENVOI 	0
#define MAITRE_RECOIT	5

#define ESCLAVE_ENVOI	MAITRE_RECOIT
#define ESCLAVE_RECOIT 	MAITRE_ENVOI

int main(argc, argv) int argc; char *argv[]; {
	/*========================================================================*/
	/* Declaration de variables et allocation memoire */
	/*========================================================================*/

	int i, j, n;
	int info ;
	int nbhost, nbtaches, nbarch;
	int msgtype ;
	int who;
	
	int LE_MIN = MAX_VALEUR;
	int LE_MAX = MIN_VALEUR;
	
	float ETALEMENT = 0.0;
	
	int **image;
	int **resultat;
	int X, Y, x, y;
	int TailleImage;

	int NbResultats, quelle_ligne, lignes;
	int *la_ligne;
	
	int P;
	
	FILE *Src, *Dst;

	char SrcFile[MAX_CHAINE];
	char DstFile[MAX_CHAINE];
	
	char ligne[MAX_CHAINE];
	
	int NumLigne, NumTache;
	int ReponsesRecues;
	
	boolean fin ;
	boolean inverse = false;
	
	char *Chemin;
	char *CheminTache;

	struct pvmhostinfo *hostp;
	struct pvmtaskinfo *taskinfo;
	char *param[MAX_PARAM];
	int numtaches[MAX_HOSTS];
	int mytid, nhost, narch;
	int iddaemon;
	
	/*========================================================================*/
	/* Recuperation des parametres						*/
	/*========================================================================*/

	sscanf(argv[1],"%s", SrcFile);
	
	sprintf(DstFile,"%s.new",SrcFile);
	
	/*========================================================================*/
	/* Recuperation de l'endroit ou l'on travail				*/
	/*========================================================================*/

	CALLOC(Chemin, MAX_CHAINE, char);
	CALLOC(CheminTache, MAX_CHAINE, char);
	Chemin = getenv("PWD");
	printf("Repertoire de travail : %s \n\n",Chemin);
	sprintf(CheminTache, "./ESCLA");

	/*========================================================================*/
	/* Ouverture des fichiers						*/
	/*========================================================================*/

	printf("Operations sur les fichiers\n");

	FOPEN(Src, SrcFile, "r");
	printf("\t Fichier source ouvert (%s) \n",SrcFile);
		
	FOPEN(Dst, DstFile, "w");
	printf("\t Fichier destination ouvert (%s) \n",DstFile);
	
	/*========================================================================*/
	/* On effectue la lecture du fichier source */
	/*========================================================================*/
	
	printf("\t Lecture entete du fichier source ");
	
	for (i = 0 ; i < 2 ; i++) {
		fgets(ligne, MAX_CHAINE, Src);	
		fprintf(Dst,"%s", ligne);
	}	

	fscanf(Src," %d %d\n",&X, &Y);
	fprintf(Dst," %d %d\n", X, Y);
	
	fgets(ligne, MAX_CHAINE, Src);	/* Lecture du 255 	*/
	fprintf(Dst,"%s", ligne);
	
	printf(": OK \n");
	
	/*========================================================================*/
	/* Allocation memoire pour l'image source et l'image resultat 		*/
	/*========================================================================*/
	
	CALLOC(image, Y+1, int *);
	CALLOC(resultat, Y+1, int *);
	for (i=0;i<Y;i++) {
		CALLOC(image[i], X+1, int);
		CALLOC(resultat[i], X+1, int);
		for (j=0;j<X;j++) {
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
	/* Lecture du fichier pour remplir l'image source 			*/
	/*========================================================================*/
	
	while (! feof(Src)) {
		n = fscanf(Src,"%d",&P);
		image[y][x] = P;	
		LE_MIN = MIN(LE_MIN, P);
		LE_MAX = MAX(LE_MAX, P);
		x ++;
		if (n == EOF || (x == X && y == Y-1)) {
			break;
		}
		if (x == X) {
			x = 0 ;
			y++;
		}
	}
	fclose(Src);
	printf("\t Lecture du fichier image : Ok \n\n");
	
	/*========================================================================*/
	/* Calcul du facteur d'etalement					*/
	/*========================================================================*/
	
	if (inverse) {
		ETALEMENT = 0.2;	
	} else {
		ETALEMENT = (float)(MAX_VALEUR - MIN_VALEUR) / (float)(LE_MAX - LE_MIN);	
	}
	
	
	
	/*========================================================================*/
	/* Calcul de cahque nouvelle valeur de pixel							*/
	/*========================================================================*/

	hostp = calloc(1, sizeof(struct pvmhostinfo));
	taskinfo = calloc(1, sizeof(struct pvmtaskinfo));

	mytid = pvm_mytid();

	// List les taches deja en cours
	info = pvm_tasks(0, &nbtaches, &taskinfo);
	for (i=0 ; i < nbtaches ; i++) {
		if (taskinfo[i].ti_tid == mytid) {
			printf("\t Tache %d : tourne sur le noeud (ti_host) : %d ; Commentaire TACHE PRINCIPALE \n", taskinfo[i].ti_tid, taskinfo[i].ti_host);
		} else {
			printf("\t Tache daemon %d : tourne sur le noeud (ti_host) : %d ;\n", taskinfo[i].ti_tid, taskinfo[i].ti_host);
			iddaemon = taskinfo[i].ti_tid;
		}
	}

	// Creation des taches !
	info = pvm_config(&nhost, &narch, &hostp);
	for (i=0 ; i < nhost ; i++) {
		printf("\tNoeud %d : \n", i);
		printf("\t\t hi_tid = %d \n",hostp[i].hi_tid);
		printf("\t\t hi_name = %s \n",hostp[i].hi_name);
		printf("\t\t hi_arch = %s \n",hostp[i].hi_arch);
		printf("\t\t hi_speed = %d \n",hostp[i].hi_speed);

		param[0] = calloc(1, MAX_CHAINE);
		param[1] = calloc(1, MAX_CHAINE);
		sprintf(param[0],"%d",i);
		sprintf(param[1],"%s",hostp[i].hi_name);
		param[2] = NULL;

		nbtaches = pvm_spawn(CheminTache, &param[0], PvmTaskHost, hostp[i].hi_name,1, &numtaches[i]);

		printf("\tLance une tache sur %s : Tache %d %s %s (%d) \n", hostp[i].hi_name, numtaches[i], param[0], param[1], nbtaches);
	}

	info = pvm_tasks(0, &nbtaches, &taskinfo);
	printf("\nListe des taches de la Parallel Virtuelle Machine : %d \n", nbtaches);
	for (i=0 ; i < nbtaches ; i++) {
		if (taskinfo[i].ti_tid == iddaemon) {
			printf("\t Tache %d : tache systeme pvmd (Daemon) \n",taskinfo[i].ti_tid);
		} else if (taskinfo[i].ti_tid == mytid) {
			printf("\t Tache %d : tourne sur le noeud (ti_host) : %d ; Commentaire TACHE PRINCIPALE \n",
				taskinfo[i].ti_tid, taskinfo[i].ti_host);
		} else {
			printf("\t Tache %d : tourne sur le noeud (ti_host) : %d ;\n", 
				taskinfo[i].ti_tid, taskinfo[i].ti_host);
		}

	}

	NumLigne = 0;

	// second loop to send ligne to each sub-task
	for (i=0 ; i < nhost ; i++) {
		msgtype = MAITRE_ENVOI;
		pvm_initsend(PvmDataDefault);
		pvm_pkint(&mytid, 1, 1);
		pvm_pkint(&LE_MIN, 1, 1); // Envoi LE_MIN
		pvm_pkfloat(&ETALEMENT, 1, 1); // Envoi de la valeur d'etalement
		pvm_pkint(&x, 1, 1); // Envoi de la taille de la ligne
		pvm_pkint(&NumLigne, 1, 1); // Envoi du num de la ligne
		pvm_pkint(image[NumLigne], X, 1); // Envoi de la ligne

		pvm_send(numtaches[i], msgtype);

		printf("\tEnvoi de la ligne %d a la tache %d \n", NumLigne, numtaches[i]);

		NumLigne++;
	}

	// Reception des resultats et envoi des autres lignes
	ReponsesRecues = 0;
	i=0;
	//for (i=NumLigne ; i < Y ; i++) { // Y = nombre de ligne
	while (ReponsesRecues < Y) {
		printf("Attente de reception\n\n");
		msgtype = MAITRE_RECOIT;

		pvm_recv(-1, msgtype);
		pvm_upkint(&who, 1, 1);
		// indice de la tache
		for (j=0 ; j < nbtaches ; i++) {
			if (mytid == taskinfo[j].ti_tid) {
				i = j;
				printf("I==%d\n", i);
			}
		}
		pvm_upkint(&quelle_ligne, 1, 1);
		if (quelle_ligne == -1) {
			pvm_upkint(&n, 1, 1);
			printf("\tLa tache %d a traiter %d lignes\n", who, n);
		} else {
			pvm_upkint(resultat[quelle_ligne], X, 1);

			ReponsesRecues++;

			msgtype = MAITRE_ENVOI;
			pvm_initsend(PvmDataDefault);
			if (NumLigne == Y) {
				n=-1;
				pvm_pkint(&n, 1, 1); // Envoi du message de fin
				pvm_send(numtaches[i], msgtype);
			} else {
				pvm_pkint(&NumLigne, 1, 1); // Envoi du num de la ligne
				pvm_pkint(image[NumLigne], X, 1); // Envoi de la ligne

				pvm_send(numtaches[i], msgtype);
				NumLigne++;

				printf("\tEnvoi de la ligne %d a la tache %d \n", NumLigne, numtaches[i]);
			}
		}
	}

	/*for (i = 0 ; i < Y ; i++) {
		for (j = 0 ; j < X ; j++) {
			resultat[i][j] = ((image[i][j] - LE_MIN) * ETALEMENT);
		}
	}*/

	pvm_exit();

	/*========================================================================*/
	/* Sauvegarde de l'image dans le fichier resultat			*/
	/*========================================================================*/
	
	n = 0;
	for (i = 0 ; i < Y ; i++) {
		for (j = 0 ; j < X ; j++) {
			
			fprintf(Dst,"%3d ",resultat[i][j]);
			n++;
			if (n == NBPOINTSPARLIGNES) {
				n = 0;
				fprintf(Dst, "\n");
			}
		}
	}
				
	fprintf(Dst,"\n");
	fclose(Dst);
	
	printf("\n");

	/*========================================================================*/
	/* Fin du programme principal	*/
	/*========================================================================*/

	exit(0); 
	
}

