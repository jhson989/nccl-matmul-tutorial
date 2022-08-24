CC = nvcc
PROGRAM = program.out
SRCS = main.cu
INCS =
OPTS = -lnccl

.PHONY : all run clean

all: ${PROGRAM}

${PROGRAM}: ${SRCS} ${INCS} Makefile
	${CC} -o $@ ${SRCS} ${OPTS}


run : ${PROGRAM}
	./${PROGRAM}

clean :
	rm ${PROGRAM}


