CC      =       cc
CFLAGS	=	-g  -lm -lpvm3
RM 	= rm -f

all: CodeSequentiel ESCLA



CodeSequentiel:	CodeSequentiel.c 
	$(CC) -o CodeSequentiel CodeSequentiel.c $(CFLAGS)   

ESCLA:	ESCLA.c 
	$(CC) -o ESCLA ESCLA.c  $(CFLAGS)  
	cp ESCLA $(HOME)

clean:
	$(RM) CodeSequentiel
	$(RM) ESCLA 
	$(RM) $(HOME)/ESCLA
