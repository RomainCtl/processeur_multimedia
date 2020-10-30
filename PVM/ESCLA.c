#include <stdlib.h>
#include <stdio.h>
#include "pvm3.h"


#define MAX_CHAINE 100

#define CALLOC(ptr, nr, type) 		if (!(ptr = (type *) calloc((size_t)(nr), sizeof(type)))) {		\
						printf("Erreur lors de l'allocation memoire \n") ; 		\
						exit (-1);							\
					} 

#define MAITRE_ENVOI 	0
#define MAITRE_RECOIT	5

#define ESCLAVE_ENVOI	MAITRE_RECOIT
#define ESCLAVE_RECOIT 	MAITRE_ENVOI

main(argc, argv) int argc; char *argv[]; {
	FILE * fp;
	fp = fopen ("/tmp/escla_log.txt", "w+");
	fprintf(fp, "===========log1");
	int i, j;
	int param;
	int indice1 = -1; // me
	int indice2 = -1; // maitre
	int mytid;
	int nbtaches;
	int info1;
	int info2;
	int ti_host;
	int nhost;
	int narch;
	int maitre;
	int msgtype;
	int newvaleur;
	char machine[MAX_CHAINE];

	int le_min;
	int ligne_num;
	float etalement;
	int ligne_size;
	int *ligne;
	int *resultat;

	int nb_ligne=0;

	struct pvmtaskinfo *taskinfo;
	struct pvmhostinfo *hostp;

	wait(2);
	fprintf(fp, "===========log2");
	
	sscanf(argv[1],"%d",&param);
	sscanf(argv[2],"%s",machine);
	
	taskinfo = calloc(1, sizeof(struct pvmtaskinfo));
	hostp = calloc(1, sizeof(struct pvmhostinfo));

	mytid = pvm_mytid();
	fprintf(fp, "===========log3");

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
	fprintf(fp, "===========log4");

	msgtype = ESCLAVE_RECOIT;
	pvm_recv(-1 , msgtype);
	pvm_upkint(&maitre, 1, 1);
	pvm_upkint(&le_min, 1, 1);
	pvm_upkfloat(&etalement, 1, 1);
	pvm_upkint(&ligne_size, 1, 1);
	fprintf(fp, "===========log5");

	do {
		CALLOC(ligne, ligne_size+1, int);
		pvm_upkint(&ligne_num, 1, 1);

		if (ligne_num == -1) {
			// Fin
			msgtype = ESCLAVE_ENVOI;
			pvm_initsend(PvmDataDefault);
			pvm_pkint(&mytid, 1, 1);
			pvm_pkint(&ligne_num, 1, 1); // Pour dire que c'est le cptrendu
			pvm_pkint(&nb_ligne, 1, 1);
			pvm_send(maitre, msgtype);
		} else {
			nb_ligne++;
			pvm_upkint(&ligne[0], ligne_size, 1);

			for (j = 0 ; j < ligne_size ; j++) {
				resultat[j] = ((ligne[j] - le_min) * etalement);
			}

			msgtype = ESCLAVE_ENVOI;
			pvm_initsend(PvmDataDefault);
			pvm_pkint(&mytid, 1, 1);
			pvm_pkint(&ligne_num, 1, 1);
			pvm_pkint(resultat, ligne_size, 1);
			pvm_send(maitre, msgtype);
		}
	} while(ligne_num != -1);


	pvm_exit();
	exit(0);
}





