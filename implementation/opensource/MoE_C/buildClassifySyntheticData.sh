gcc -c MoE.c -o MoE.o
gcc -c ClassifySyntheticData.c -o ClassifySyntheticData.o
gcc MoE.o ClassifySyntheticData.o -lm -o ClassifySyntheticData