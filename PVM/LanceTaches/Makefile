CC      =       cc
CFLAGS	=	-g  -lm -lpvm3
RM 		= rm -f
all: LanceTaches Tache 



LanceTaches:	LanceTaches.c 
	$(CC) -o LanceTaches LanceTaches.c $(CFLAGS)   

Tache:	Tache.c 
	$(CC)  -o Tache Tache.c  $(CFLAGS)  
	cp Tache $(HOME)

clean:
	$(RM) LanceTaches 
	$(RM) Tache 
	$(RM) $(HOME)/Tache







