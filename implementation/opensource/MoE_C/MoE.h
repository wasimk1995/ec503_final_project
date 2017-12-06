//      MoE.h
//      
//      Copyright 2011 Goker Erdogan <goker@goker-laptop>
//      
//      This program is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//      
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//      
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//      MA 02110-1301, USA.

/*
* Header file containing Mixture of Experts data structures
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

enum MoEClusteringType
{
	COOPERATIVE,
	COMPETITIVE
};


struct MoEIntermediateCalculations
{
	double *g;
	double *w;
	double *y;
	double *o;
	double *f;
	double *yih;
};

typedef struct MoEIntermediateCalculations MoEIntermediateCalculationsType;

struct MoE
{
	int inputDimension;
	int outputDimension;
	int expertCount;
	enum MoEClusteringType clusteringType;
	double learningRate;
	double decay;
	// weight vectors of each expert for each output
	double *v;
	// centers of each expert
	double *m;	
	// intermediate calculations
	MoEIntermediateCalculationsType imCalc;
};

typedef struct MoE MoEType;


/* InitializeMoE: initializes MoE data structure by allocating storage for 
* parameters and initializing these parameters
*
*/
MoEType* InitializeMoE( int inputDimension, int outputDimension, int expertCount, enum MoEClusteringType clType, double learningRate, double decay);

/* TrainOnlineD: updates MoE parameters according to given sample
*
*/
void TrainOnlineD( MoEType *moe, double *x, int r, int epochFinished);

void TrainOnlineWithSparseInput( MoEType *moe, int *indices, int *counts, int length, int r, int epochFinished);

/* TestSample: tests sample with MoE and returns class label
*
*/
int TestSample( MoEType *moe, double *x);

int TestSampleWithSparseInput( MoEType *moe, int *indices, int *counts, int length);

/* SaveMoEParameters: Saves m and v matrices to file
*
*/
void SaveMoEParameters( MoEType *moe, char* mFileName, char* vFileName);

/* PrintMoE: Prints info about MoE
*
*/
void PrintMoE( MoEType *moe);
/* FreeMoE: frees memory allocated for MoE
*
*/
void FreeMoE( MoEType *moe );

