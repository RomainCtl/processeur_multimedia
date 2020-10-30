/*==============================================================================*/
/* Programme 	: ConfPVM.c							*/
/* Auteur 	: Daniel CHILLET						*/
/* Date 	: Novembre 2001							*/
/* 										*/
/* Objectifs	: Programme principal permet de faire une analyse rapide 	*/
/*		de l'etat de la machine virtuelle PVM				*/
/* 										*/
/* Fonctionnemnt: 1) Demande la configuration de la machine PVM			*/
/*		2) Effectue l'affichage des hosts de la machine PVM 		*/
/*		3) Effectue l'affichage des tacches de la machine PVM 	*/
/*==============================================================================*/



#include <stdlib.h>
#include <stdio.h>
#include "pvm3.h"

#define RESULT_TAG 1
#define MAX_LENGTH 128



main(argc, argv)
int argc;
char *argv[];
{
	int i;
	int info ;
	int mytid;
	int nhost;
	int narch;
	int nbtaches;
	struct pvmhostinfo *hostp;
	struct pvmtaskinfo *taskinfo;
	
	hostp = calloc(1, sizeof(struct pvmhostinfo));
	
	info = pvm_config(&nhost, &narch, &hostp);

	printf("Nombre de noeuds dans la Parallel Virtual Machine : %d\n",nhost);
	printf("Nombre d'architecture dans la Parallel Virtual Machine : %d\n",narch);
	
	printf("\nListe des noeuds de la PVM : \n");
	for (i=0 ; i < nhost ; i++) {
		printf("\tNoeud %d : \n", i);
		printf("\t\t hi_tid = %d \n",hostp[i].hi_tid);
		printf("\t\t hi_name = %s \n",hostp[i].hi_name);
		printf("\t\t hi_arch = %s \n",hostp[i].hi_arch);
		printf("\t\t hi_speed = %d \n",hostp[i].hi_speed);

	}


	taskinfo = calloc(1, sizeof(struct pvmtaskinfo));

	mytid = pvm_mytid();
	
	info = pvm_tasks(0, &nbtaches, &taskinfo);
	printf("\nNombre de taches tournant dans la PVM : %d\n",nbtaches);
	printf("\nListe des taches de la PVM : \n");
	for (i=0 ; i < nbtaches ; i++) {
		if (taskinfo[i].ti_tid == 0) {
			printf("\t Tache %d : tache systeme pvmd (Daemon) \n",taskinfo[i].ti_tid);
		} else if (taskinfo[i].ti_tid == mytid) {
			printf("\t Tache %d : tourne sur le noeud (ti_host) : %d ; Commentaire TACHE PRINCIPALE \n",
				taskinfo[i].ti_tid, taskinfo[i].ti_host);
		} else {
			printf("\t Tache %d : tourne sur le noeud (ti_host) : %d ;\n", 
				taskinfo[i].ti_tid, taskinfo[i].ti_host);
		}

	}
	printf(" \n");


	
	pvm_exit(); 
	exit(0); 
}
