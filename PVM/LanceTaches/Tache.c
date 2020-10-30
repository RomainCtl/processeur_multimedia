/*==============================================================================*/
/* Programme 	: Tache.c (Escalve)						*/
/* Auteur 	: Daniel CHILLET						*/
/* Date 	: Novermbre 2001						*(
/* 										*/
/* Objectifs	: Une tache qui fait pas grand chose d'autre que d'afficher	*/
/* 		sur quelle machine elle tourne 				*/
/* 										*/
/* Principe	: Tache est lance par le maitre (LanceTaches) 			*/
/* 										*/
/*==============================================================================*/

#include <stdlib.h>
#include <stdio.h>
#include "pvm3.h"


#define MAX_CHAINE 100

main(argc, argv)
int argc;
char *argv[];
{
	int i;
	int param;
	int indice1 = -1;
	int indice2 = -1;
	int mytid;
	int nbtaches;
	int info1;
	int info2;
	int ti_host;
	int nhost;
	int narch;
	char machine[MAX_CHAINE];
	struct pvmtaskinfo *taskinfo;
	struct pvmhostinfo *hostp;

	wait(2);
	
	sscanf(argv[1],"%d",&param);
	sscanf(argv[2],"%s",&machine);
	
	taskinfo = calloc(1, sizeof(struct pvmtaskinfo));
	hostp = calloc(1, sizeof(struct pvmhostinfo));

	mytid = pvm_mytid();

	info1 = pvm_tasks(mytid, &nbtaches, &taskinfo);
	for (i=0 ; i < nbtaches ; i++) {
		if (mytid == taskinfo[i].ti_tid) {
			indice1 = i;
		}
	}
	ti_host = taskinfo[indice1].ti_host;
	
	
	info2 = pvm_config(&nhost, &narch, &hostp);
	for (i=0 ; i < nhost ; i++) {
		if (ti_host == hostp[i].hi_tid) {
			indice2 = i;
		}
	}
	printf("Tache %d (%d): je tourne sur la machine %s (%s : %d)\n", 
		param, mytid, hostp[indice2].hi_name, machine, ti_host);
		
		
	pvm_exit(); 
	exit(0); 

}
