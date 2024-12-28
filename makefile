OLD  := MMult2
NEW  := MMult3
#
# sample makefile
#

CC         := gcc
LINKER     := $(CC)
CFLAGS     := -O3 -march=native -mavx2 -mfma -ffast-math -funroll-loops -Wall -fopenmp
LDFLAGS    := -lm -fopenmp

UTIL       := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              REF_MMult.o \
              print_matrix.o

TEST_OBJS  := test_MMult.o $(NEW).o 

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: 
	make clean;
	make test_MMult.x

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

run:	
	make all
	export OMP_NUM_THREADS=1
	export GOTO_NUM_THREADS=1
	echo "version = '$(NEW)';" > output_$(NEW).m
	./test_MMult.x >> output_$(NEW).m
	cp output_$(OLD).m output_old.m
	cp output_$(NEW).m output_new.m

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png

vtune: test_MMult.x
	vtune -collect hotspots ./test_MMult.x

vtune-advanced: test_MMult.x
	vtune -collect advanced-hotspots ./test_MMult.x

vtune-modern: test_MMult.x
	vtune -collect performance-snapshot ./test_MMult.x

test_320.x: test_320.o $(NEW).o dclock.o
	$(LINKER) test_320.o $(NEW).o dclock.o $(LDFLAGS) -o $@

test_320: test_320.x
	./test_320.x
