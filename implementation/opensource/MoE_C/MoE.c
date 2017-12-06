//      MoE.c
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
 * Functions for training a mixture of experts model for classification
 */
 
 #include "MoE.h"
 
 int MoEmain(int argc, char **argv)
 {
	return 0;
 }
 
 MoEType* InitializeMoE( int inputDimension, int outputDimension, int expertCount, enum MoEClusteringType clType, double learningRate, double decay)
 {
	MoEType *moe = malloc( sizeof(MoEType) );
	moe->inputDimension = inputDimension;
	moe->outputDimension = outputDimension;
	moe->expertCount = expertCount;
	moe->learningRate = learningRate;
	moe->decay = decay;
	moe->clusteringType = clType;
	
	// v holds coefficients for linear fits of each expert for each output (bias unit is also added)
	moe->v = malloc( sizeof(double) * (moe->outputDimension) * (moe->expertCount) * (moe->inputDimension+1) );
	
	// local region centers
	moe->m = malloc( sizeof(double) * (moe->inputDimension) * (moe->expertCount) );
	
	// allocate storage for intermediate calculation results
	// allocate storage for weight of each expert, g_h
	moe->imCalc.g = malloc( sizeof(double) * moe->expertCount );
	// allocate storage for output of each expert for each output
	moe->imCalc.w = malloc( sizeof(double) * (moe->outputDimension) * (moe->expertCount) );	
	//y_i: sum(w_ih * g_h) output for each output dimension
	//o_i: softmax(y_i)
	moe->imCalc.y = malloc( sizeof(double) * moe->outputDimension);
	moe->imCalc.o = malloc( sizeof(double) * moe->outputDimension);
	// f: used in competitive MoE updates, size of expertCount
	moe->imCalc.f = malloc( sizeof(double) * moe->expertCount );
	// y_ih: softmax(w_ih)
	moe->imCalc.yih = malloc( sizeof(double) * (moe->outputDimension) * (moe->expertCount) );	
	
	srand(time(NULL));
	
	int i, h, j;
	for( i = 0; i < moe->outputDimension; i++)
	{
		for( h = 0; h < moe->expertCount; h++ )
		{			
			for( j = 0; j < moe->inputDimension+1; j++ )
			{
				// initialize v values with random values from [-0.01, 0.01] interval
				moe->v[ (j * moe->outputDimension * moe->expertCount) + (i * moe->expertCount) + h ] = ((rand() / ( (double)RAND_MAX )) * 0.02) - 0.01;			
				
				//initialize m
				// m matrice is of inputDimension x expertCount size (no bias unit)
				if( i == 0 && j < moe->inputDimension )
				{
					moe->m[ (j * moe->expertCount) + h ] = ((rand() / ( (double)RAND_MAX )) * 0.02) - 0.01;
				}
			}			
		}
	}
	
	return moe;
	
 }
 
 void __calculateG( MoEType *moe, double *x)
 {
	int i, h, j;
	
	double *g = moe->imCalc.g;
	
	// calculate g_h = m_h x
	double totalG = 0.0;
	for( h = 0; h < moe->expertCount; h++ )
	{
		g[h] = 0.0;
		for( j = 0; j < moe->inputDimension; j++ )
		{
			g[h] += moe->m[ (j * moe->expertCount) + h ] * x[j];
		}
		//printf("g_%d: %f\n", h, g[h]);
		// softmax pt.1
		g[h] = exp(g[h]);
		totalG += g[h];
	}
	//softmax pt.2
	for( h = 0; h < moe->expertCount; h++ )
	{
		g[h] = g[h] / totalG;
	}
 }
void __calculateW( MoEType *moe, double *x)
 {
	int i, j, h;
	
	double *w = moe->imCalc.w;
	
	//calculate w
	double cx;
	for( i = 0; i < moe->outputDimension; i++)
	{
		for( h = 0; h < moe->expertCount; h++ )
		{
			w[ (i * moe->expertCount) + h ] = 0.0; 
			for( j = 0; j < moe->inputDimension+1; j++ )
			{
				if ( j == 0 ) // first input is bias unit and always 1
				{
					cx = 1;
				}
				else
				{
					cx = x[j-1];
				}
				w[ (i * moe->expertCount) + h ] += moe->v[ (j * moe->outputDimension * moe->expertCount) + (i * moe->expertCount) + h ] * cx;
			}
		}
	}
 }
 
 void __calculateYandO( MoEType *moe)
 {
	int i, h;
	
	double *y = moe->imCalc.y;
	double *o = moe->imCalc.o;
	double *w = moe->imCalc.w;
	double *g = moe->imCalc.g;
	double totalO = 0.0;
	for( i = 0; i < moe->outputDimension; i++)
	{
		y[i] = 0.0;
		for( h = 0; h < moe->expertCount; h++ )
		{			
			y[i] += w[ (i * moe->expertCount) + h ] * g[h];
		}
		//printf("y_%d: %f\n", i, y[i]);
		o[i] = exp(y[i]);
		totalO += o[i];
	}
	for( i = 0; i < moe->outputDimension; i++)
	{
		o[i] = o[i] / totalO;
	}
 }
 
 void __calculateFandYih( MoEType *moe, int r)
 {
	int i, h;
	// calculate y_ih = softmax(w_ih)
	
	// calculate totals of each expert for all outputs
	double *totalOutputs = malloc( sizeof(double) * moe->expertCount );
	for( i = 0; i < moe->outputDimension; i++)
	{		
		for( h = 0; h < moe->expertCount; h++ )
		{
			// first initialize to 0
			if ( i == 0 )
			{
				totalOutputs[h] = 0.0;
			}
			totalOutputs[h] += exp(moe->imCalc.w[ (i * moe->expertCount) + h ]);
		}		
	}
	
	/*
	for( h = 0; h < moe->expertCount; h++ )
	{
		printf("to_%d: %f ", h, totalOutputs[h]);
	}
	printf("\n");
	*/
	
	// calculate y_ih by normalization
	// NOTE: WE ONLY NEED y_ih for i where r_i = 1
	for( h = 0; h < moe->expertCount; h++ )
	{
		moe->imCalc.yih[ ((r-1) * moe->expertCount) + h ] = exp( moe->imCalc.w[ ((r-1) * moe->expertCount) + h ] ) / (totalOutputs[h]);
	}
	
	// calculate f_h: [g_h * exp( sum_i( r_i * log(y_ih) ) )] / [sum_k( g_h * exp( sum_i( r_i * log(y_ik) ) ) )]
	// since only one r_i is 1 for each sample exp ( sum_i( r_i * log(y_ih) ) ) = exp(log(y_ih)) = y_ih (where i is indice of r_i=1)
	double totalF = 0.0;
	for( h = 0; h < moe->expertCount; h++ )
	{
		moe->imCalc.f[h] = moe->imCalc.g[h] * moe->imCalc.yih[ ((r-1) * moe->expertCount) + h ];
		//printf("f_%d: %f\n", h, moe->imCalc.f[h]);
		totalF += moe->imCalc.f[h];
	}
	// normalize f_h
	for( h = 0; h < moe->expertCount; h++ )
	{
		moe->imCalc.f[h] = moe->imCalc.f[h] / totalF;		
	}
	
 }
 
 void TrainOnlineD( MoEType *moe, double *x, int r, int epochFinished)
 {
	int i, h, j;
	
	// calculate g
	__calculateG(moe, x);
	//calculate w
	__calculateW(moe, x);
	
	if( moe->clusteringType == COOPERATIVE )
	{
		__calculateYandO(moe);
		
		//update v and m
		double dv, dm;
		int ri;
		double cx;
		for( i = 0; i < moe->outputDimension; i++)
		{
			// convert class label to membership value
			ri = 0;
			if( r == i+1 )
			{
				ri = 1;
			}
			
			for( h = 0; h < moe->expertCount; h++ )
			{			
				for( j = 0; j < moe->inputDimension+1; j++ )
				{
					// first input is bias unit
					if( j == 0 )
					{
						cx = 1;
					}
					else
					{
						cx = x[j-1];
					}
					// update v
					dv = moe->learningRate * (ri - moe->imCalc.o[i]) * moe->imCalc.g[h] * cx;
					moe->v[ (j * moe->outputDimension * moe->expertCount) + (i * moe->expertCount) + h ] += dv;
					// update m
					if ( i == 0 && j < moe->inputDimension ) // m is independent of output, only in first output iteration it is updated
					{
						dm = moe->learningRate * (ri - moe->imCalc.o[i]) * (moe->imCalc.w[ (i * moe->expertCount) + h ] - moe->imCalc.y[i]) * moe->imCalc.g[h] * x[j];
						moe->m[ (j * moe->expertCount) + h ] += dm;
					}
				}
			}
		}
	}
	else // COMPETITIVE clustering
	{
		__calculateFandYih(moe, r);
		
		//update v and m
		double dv, dm;
		int ri;
		double cx;
		for( i = 0; i < moe->outputDimension; i++)
		{
			// convert class label to membership value
			ri = 0;
			if( r == i+1 )
			{
				ri = 1;
			}
			
			for( h = 0; h < moe->expertCount; h++ )
			{			
				for( j = 0; j < moe->inputDimension+1; j++ )
				{
					// first input is bias unit
					if( j == 0 )
					{
						cx = 1;
					}
					else
					{
						cx = x[j-1];
					}
					// update v
					dv = moe->learningRate * (ri - moe->imCalc.yih[ (i * moe->expertCount) + h ]) * moe->imCalc.f[h] * cx;
					moe->v[ (j * moe->outputDimension * moe->expertCount) + (i * moe->expertCount) + h ] += dv;
					// update m
					if ( i == 0 && j < moe->inputDimension ) // m is independent of output, only in first output iteration it is updated
					{
						dm = moe->learningRate * ( moe->imCalc.f[h] - moe->imCalc.g[h] ) * x[j];
						moe->m[ (j * moe->expertCount) + h ] += dm;
					}
				}
			}
		}
		
	}
	
	// if end of epoch update learning rate
	if ( epochFinished > 0 )
	{
		moe->learningRate *= moe->decay;
	}
	
 }
 
 int TestSample( MoEType *moe, double *x)
 {
	__calculateG(moe, x);
	__calculateW(moe, x);
	__calculateYandO(moe);
	
	int i;
	double maxy = 0;
	int classLabel;
	// find class label with maximum output value
	for( i = 0; i < moe->outputDimension; i++)
	{
		//printf("o_%d: %f ,", i, moe->imCalc.o[i]);
		if ( moe->imCalc.o[i] > maxy )
		{
			maxy = moe->imCalc.o[i];
			classLabel = i+1;
		}
	}
	return classLabel;
 }
 
 /*************Functions for Sparse Input*******************/
 void __calculateWFromSparseInput( MoEType *moe, int *indices, int *counts, int length)
 {
	int i, j, h;
	
	double *w = moe->imCalc.w;
	
	//calculate w
	double cx;
	for( i = 0; i < moe->outputDimension; i++)
	{
		for( h = 0; h < moe->expertCount; h++ )
		{
			w[ (i * moe->expertCount) + h ] = 0.0; 
			for( j = 0; j < length; j++ )
			{				
				w[ (i * moe->expertCount) + h ] += moe->v[ (indices[j] * moe->outputDimension * moe->expertCount) + (i * moe->expertCount) + h ] * counts[j];
			}
		}
	}
 }
 
  void __calculateGFromSparseInput( MoEType *moe, int *indices, int *counts, int length)
 {
	int i, h, j;
	
	double *g = moe->imCalc.g;
	
	// calculate g_h = m_h x
	double totalG = 0.0;
	for( h = 0; h < moe->expertCount; h++ )
	{
		g[h] = 0.0;
		for( j = 0; j < length; j++ )
		{
			g[ h ] += moe->m[ (indices[j] * moe->expertCount) + h ] * counts[j];
		}
		//printf("g_%d: %f\n", h, g[h]);
		// softmax pt.1
		g[h] = exp(g[h]);
		totalG += g[h];
	}
	//softmax pt.2
	for( h = 0; h < moe->expertCount; h++ )
	{
		g[h] = g[h] / totalG;
	}
 }
 
 void TrainOnlineWithSparseInput( MoEType *moe, int *indices, int *counts, int length, int r, int epochFinished)
 {
	int i, h, j;
	
	// calculate g
	__calculateGFromSparseInput(moe, indices, counts, length);
	//calculate w
	__calculateWFromSparseInput(moe, indices, counts, length);
	
	if( moe->clusteringType == COOPERATIVE )
	{
		__calculateYandO(moe);
		
		//update v and m
		double dv, dm;
		int ri;
		double cx;
		for( i = 0; i < moe->outputDimension; i++)
		{
			// convert class label to membership value
			ri = 0;
			if( r == i+1 )
			{
				ri = 1;
			}
			
			for( h = 0; h < moe->expertCount; h++ )
			{			
				for( j = 0; j < length; j++ )
				{					
					// update v
					dv = moe->learningRate * (ri - moe->imCalc.o[i]) * moe->imCalc.g[h] * counts[j];
					moe->v[ (indices[j] * moe->outputDimension * moe->expertCount) + (i * moe->expertCount) + h ] += dv;
					// update m
					if ( i == 0 ) // m is independent of output, only in first output iteration it is updated
					{
						dm = moe->learningRate * (ri - moe->imCalc.o[i]) * (moe->imCalc.w[ (i * moe->expertCount) + h ] - moe->imCalc.y[i]) * moe->imCalc.g[h] * counts[j];
						moe->m[ (indices[j] * moe->expertCount) + h ] += dm;
					}
				}
			}
		}
	}
	else // COMPETITIVE clustering
	{
		__calculateFandYih(moe, r);
		
		//update v and m
		double dv, dm;
		int ri;
		double cx;
		for( i = 0; i < moe->outputDimension; i++)
		{
			// convert class label to membership value
			ri = 0;
			if( r == i+1 )
			{
				ri = 1;
			}
			
			for( h = 0; h < moe->expertCount; h++ )
			{			
				for( j = 0; j < length; j++ )
				{					
					// update v
					dv = moe->learningRate * (ri - moe->imCalc.yih[ (i * moe->expertCount) + h ]) * moe->imCalc.f[h] * counts[j];
					moe->v[ (indices[j] * moe->outputDimension * moe->expertCount) + (i * moe->expertCount) + h ] += dv;
					// update m
					if ( i == 0 ) // m is independent of output, only in first output iteration it is updated
					{
						dm = moe->learningRate * ( moe->imCalc.f[h] - moe->imCalc.g[h] ) * counts[j];
						moe->m[ (indices[j] * moe->expertCount) + h ] += dm;
					}
				}
			}
		}
		
	}
	
	// if end of epoch update learning rate
	if ( epochFinished > 0 )
	{
		moe->learningRate *= moe->decay;
	}
	
 }
 
 int TestSampleWithSparseInput( MoEType *moe, int *indices, int *counts, int length)
 {
	__calculateGFromSparseInput(moe, indices, counts, length);
	__calculateWFromSparseInput(moe, indices, counts, length);
	__calculateYandO(moe);
	
	int i;
	double maxy = 0;
	int classLabel;
	// find class label with maximum output value
	for( i = 0; i < moe->outputDimension; i++)
	{
		//printf("o_%d: %f ,", i, moe->imCalc.o[i]);
		if ( moe->imCalc.o[i] > maxy )
		{
			maxy = moe->imCalc.o[i];
			classLabel = i+1;
		}
	}
	return classLabel;
 }
 
 /**********************************************************/
 
 void SaveMoEParameters( MoEType *moe, char* mFileName, char* vFileName)
 {
	FILE *fp = fopen(mFileName, "wb");
	fwrite(moe->m, sizeof(double), (moe->inputDimension) * (moe->expertCount), fp);
	fclose(fp);
	fp = fopen(vFileName, "wb");
	fwrite(moe->v, sizeof(double), (moe->outputDimension) * (moe->expertCount) * (moe->inputDimension+1), fp);
	fclose(fp);
 }
 
 void PrintMoE( MoEType *moe)
 {
	printf("MoE Info\n");
	printf("Input Dimension: %d Output Dimension: %d Expert Count: %d\n", moe->inputDimension, moe->outputDimension, moe->expertCount);
	if( moe->clusteringType == COOPERATIVE )
	{
		printf("Clustering Type: Cooperative ");
	}
	else
	{
		printf("Clustering Type: Competitive ");
	}
	printf("Learning Rate: %f Decay Coefficient: %f\n", moe->learningRate, moe->decay);
 }
 
 void PrintMoEToFile( MoEType *moe, FILE *fp)
 {
	fprintf(fp, "MoE Info\n");
	fprintf(fp, "Input Dimension: %d Output Dimension: %d Expert Count: %d\n", moe->inputDimension, moe->outputDimension, moe->expertCount);
	if( moe->clusteringType == COOPERATIVE )
	{
		fprintf(fp, "Clustering Type: Cooperative ");
	}
	else
	{
		fprintf(fp, "Clustering Type: Competitive ");
	}
	fprintf(fp, "Learning Rate: %f Decay Coefficient: %f\n", moe->learningRate, moe->decay);
 }
 
 void FreeMoE(MoEType* moe)
 {
	free(moe->v);
	free(moe->m);
	free(moe->imCalc.g);
	free(moe->imCalc.y);
	free(moe->imCalc.w);
	free(moe->imCalc.o);
	free(moe);
 }
 