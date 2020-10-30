/*==============================================================================*/
/* Programme 	: LanceTaches.c	(Maitre)					*/
/* Auteur 	: Daniel CHILLET						*/
/* Date 	: Novembre 2001							*/
/* 										*/
/* Objectifs	: Programme principal permettant de lancer 1 tache par		*/
/*		  host de la machine virtuelle					*/
/* 										*/
/* Principe	: Ce programme est base sur un calcul distribue sur une machine	*/
/*		  PVM. 								*/
/* 										*/
/* Fonctionnemnt: 1) Analyse des hosts de la machine virtuelle			*/
/*		  2) Lancement de 1 tache par host 	 			*/
/*		  3) Liste des taches de la machine PVM	 			*/
/*==============================================================================*/


#include <stdlib.h>
#include <stdio.h>
#include "pvm3.h"

#define RESULT_TAG 1
#define MAX_LENGTH 128

#define MAX_HOSTS 100
#define MAX_CHAINE 100
#define MAX_PARAM 10

#define MAITRE_ENVOI 	0
#define MAITRE_RECOIT	5

#define ESCLAVE_ENVOI	MAITRE_RECOIT
#define ESCLAVE_RECOIT 	MAITRE_ENVOI
main(argc, argv)
int argc;
char *argv[];
{
	int info ;
	int nhost;
	int mytid;
	int who, retour;

	int narch;
	struct pvmhostinfo *hostp;
	struct pvmtaskinfo *taskinfo;
	int i;
	int numtaches[MAX_HOSTS];
	int nbtaches;
	char *param[MAX_PARAM];

	char *Chemin[MAX_CHAINE];
	char Fichier[MAX_CHAINE];

	*Chemin = getenv("PWD");
/*	sprintf(Fichier, "%s/Tache",*Chemin);*/
	sprintf(Fichier, "./Tache");

	hostp = calloc(1, sizeof(struct pvmhostinfo));
	taskinfo = calloc(1, sizeof(struct pvmtaskinfo));
	
	int iddaemon ;
	int msgtype; 

	int valeur;

	mytid = pvm_mytid();


	info = pvm_tasks(0, &nbtaches, &taskinfo);
	printf("\nListe des taches de la Parallel Virtuelle Machine \n");
	for (i=0 ; i < nbtaches ; i++) {
		if (taskinfo[i].ti_tid == mytid) {
			printf("\t Tache %d : tourne sur le noeud (ti_host) : %d ; Commentaire TACHE PRINCIPALE \n",
				taskinfo[i].ti_tid, taskinfo[i].ti_host);
		} else {
			printf("\t Tache daemon %d : tourne sur le noeud (ti_host) : %d ;\n", 
				taskinfo[i].ti_tid, taskinfo[i].ti_host);
			iddaemon = taskinfo[i].ti_tid;
		}

	}

	info = pvm_config(&nhost, &narch, &hostp);

	printf("Nombre de noeuds dans la Parallel Virtual Machine : %d\n",nhost);
	printf("Nombre d'architecture dans la Parallel Virtual Machine : %d\n",narch);
	
	printf("\nListe des machines de la PVM :  \n");
	for (i=0 ; i < nhost ; i++) {
		printf("\tNoeud %d : \n", i);
		printf("\t\t hi_tid = %d \n",hostp[i].hi_tid);
		printf("\t\t hi_name = %s \n",hostp[i].hi_name);
		printf("\t\t hi_arch = %s \n",hostp[i].hi_arch);
		printf("\t\t hi_speed = %d \n",hostp[i].hi_speed);

		param[0] = calloc(1, MAX_CHAINE);
		param[1] = calloc(1, MAX_CHAINE);
		sprintf(param[0],"%d",i);
		sprintf(param[1], "%s",hostp[i].hi_name); 
		param[2] = NULL;
		
		/* Pour lancer la tache sur un host particulier,
		il faut placer le flag PvmTaskHost 
		Sinon, le systeme choisit tout seul le meilleur host
		pour lancer la tache */
		  
      		nbtaches = pvm_spawn(Fichier, &param[0], PvmTaskHost, hostp[i].hi_name,1, &numtaches[i]);
		
		printf("\tLance une tache sur %s : Tache %d %s %s (%d) \n", 
			hostp[i].hi_name, numtaches[i], param[0], param[1], nbtaches);
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


	valeur = mytid;

	printf("Valeur envoyee de %d : valeur envoyee : %d \n", mytid, valeur);

	msgtype = MAITRE_ENVOI;
	pvm_initsend(PvmDataDefault);
	pvm_pkint(&valeur, 1, 1);
	pvm_send(numtaches[0], msgtype);
		
	printf("Envoie effectue \n");


	printf("En reception ... \n");
	msgtype = MAITRE_RECOIT;

	pvm_recv(-1, msgtype);
	pvm_upkint(&who, 1, 1);
	pvm_upkint(&retour, 1, 1);


	printf("Message recu de %d : valeur recue : %d \n", who, retour);

	pvm_exit(); 
	exit(0); 
}

