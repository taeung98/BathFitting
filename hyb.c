#include "main.h"
#include "gsl.h"
//#include "save_h5.h"
#include "malloc.h"
#include "legendre_poly.h"
#define Nmax	8192
#define B		128
#define ETA		0.15
#define PI		M_PI
#define L		128
#define Amax	128
#define Wn(n, B)	((2*(n)+1)*M_PI/(B))
#define Wr(n, N) (-8. + (n)*16./(N-1)) // e_min=-8, e_max=8
#define N_EIGEN 1024
#define N_DOS 1024
#define epsabs 1e-7
#define epsrel 1e-7
#define lim		256	
#define stepsize 1e-5	// Initial step size for CG

#define Repeat	3.6e5	

#define Nb 17
#define CONCAT_Nb(x) "BATH_FIT_N" #x ".h5"
#define FILE_h5 CONCAT_Nb(Nb)

int my_rank, my_size;
double  ti[L], wi[L];	//Gaussian node, weight
char *H;	//Hamiltonian 
double mu;		double *wn, *w, *nth;		
double complex *Giwn, *Diwn, *Gw, *Dw, *Gt, *Dt, *Dt_re, *Rhyb_i, *Rhyb_f, *Ihyb_i, *Ihyb_f, *An, *Gt_re;
double Gre( double freq, void* params ), Gim(double freq, void* params );
struct E_params { double n; double m; double h; int N;};
struct G_params { int n; double complex *w; struct E_params e_params;};
double E( double, void* );
double obj_f( const gsl_vector*, void* );
void obj_df(const gsl_vector*, void*, gsl_vector* );
void obj_fdf( const gsl_vector*, void*, double *,  gsl_vector* );
void calculate_GD( double* freq, double complex* G, double complex* hybridization, void* params, char domain );
void FT( double complex* func_i,  double* freq_i,  double complex* func_f,  double* freq_f, double cycle);
void AN( double complex* func, double* x, double complex* coeff,  double* Gaussian_node, double* Gaussian_weight, char* fname );
void LT( double* Gaussian_node, double complex* coeff , double complex* trasform_func, int num_of_coeff );
void inverse( double* Gaussian_node,  double complex* coeff,  double complex* re_func, char* folder);
void Gaussian( double init_node,  double finl_node, int num_of_interval,  double* Gaussian_node,  double* Gaussian_weight );
void hyb( double complex* hybridization,  double* bath, char domain);	
void CGM( double* bath, double* gather_chi, int iteration );	//Conjugate Gredient Method
void fermiNdos( void* parmas );
double bisection( double desired_value, double* Energy, double* Sum_of_DOS, int size); 
double derivative( double* x, double complex* y, int index, int size);
int find_extrema( double* x, double complex* y, int* save_index);
//void Error( double* x, double complex* standard_func, double complex* func1, double complex* func2, int size, char* fname);
void pathname( void* prams);
void init_bath( double* bath, int interation, int* extrema_index, int* half_index, int peak_num);
void fitting( void* params);
double integral_imag( double* x, double complex* y, int size);
void generate_V( double* bath, int iteration, double error_range);
void generate_ep( double* bath, int iteration, int* extrema_index, int* half_index, int peak_num );
double generate_randnum( double standard_value, double err_range );
void select_hop( struct G_params*, int, int, int, int, int, int, int, int );
double abs_sign( double value);
void call_bath( char* recall_file_name, double* recall_bath);
void FWHM(double* x, double complex* y, int* peak_index, int num_of_peak, int index_of_minimum, int* half_index );
void randSplit( int n, int r, int* save_arr ); // one of nCr combinations
void ArrToStr( int* num_of_epsilon_round_the_peak, int peak_num,  char* fname, int size,  int used_pointer_address );
void swap_double(double* a, double* b);
void swap_int(int* a, int* b);
int partition(double* arr, int* indices, int low, int high);
void quickSort(double* arr, int* indices, int low, int high);
void save_bath(double* data, double* chi, int order);

int main(){
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &my_size);
	
	H = mchar(1024);
	struct G_params* p = (struct G_params*)malloc(sizeof(struct G_params));		p->w = mdcomplex(Nmax);
	w = mdouble(Nmax);			wn = mdouble(Nmax); 
	Giwn = mdcomplex(Nmax); 	Diwn = mdcomplex(Nmax);	
	Gw = mdcomplex(Nmax);		Dw = mdcomplex(Nmax);		Dt = mdcomplex(Nmax); Dt_re = mdcomplex(Nmax);
	Gt = mdcomplex(Nmax);		An = mdcomplex(Amax);		nth = mdouble(L); 
	Gt_re = mdcomplex(Nmax);	Rhyb_i = mdcomplex(Nmax);	Rhyb_f = mdcomplex(Nmax);//R_domain
								Ihyb_i = mdcomplex(Nmax);	Ihyb_f = mdcomplex(Nmax);//I_domain
	select_hop(p, 3, 4, 16, 17, 1, 2, 10, 11);
//  ->	select_hop(p, 3, 20, 95, 5);
	free(w); free(wn); free(p->w); free(Giwn); free(Diwn); free(Gw); free(Dw), free(Gt);
	free(An); free(nth); free(Gt_re); free(H); free(p); free(Rhyb_i); free(Rhyb_f); free(Dt);
	free(Dt_re); free(Ihyb_i); free(Ihyb_f); 
		
	MPI_Finalize();
	return 0; 
}

void select_hop(struct G_params *p,int Ni, int Nf, int ni, int nf, int mi, int mf, int hi, int hf){
	for(int N=Ni; N<Nf; N++){
		p->e_params.N = N;
		if( N == 1){ fitting(p);}
		else if(  N != 1 ){
			for(int i=ni; i<nf; i++){
				p->e_params.n = 1. - .05*i;
				if( N == 2){ fitting(p); }
				if( N != 2 ){
					for(int j=mi; j<mf; j++){
						p->e_params.m = ( 1. - .05*j ); 
						if( N == 3){ fitting(p); }
						if( N != 3){
							for(int z=hi; z<hf; z++){
								p->e_params.h = ( 1. - .05*z);
								if(N == 4){ fitting(p); }
							}
						}
					}
				}
			}
		}
	}
}

void fitting(void* params){
	struct G_params* p = (struct G_params*)params;
	pathname(p);	/* detrmine neighbor */		
	fermiNdos(p);	// calculate fermi value & DOS	
	// if 3rd value is 'r',calculate G in real domain, else in imag domain 	
	calculate_GD(w, Gw, Dw, p, 'r');			
	calculate_GD(wn, Giwn, Diwn, p, 'i'); 
	Gaussian(0, B, L, ti, wi);
	//for Green's function	
	FT(Giwn, wn, Gt, ti, B);	/*(save_data(ti, Gt, L, H, "Gt"); */	
	AN(Gt, nth, An, ti, wi, "Gt");	//save_data(nth, An, Amax, H, "An");
	inverse(ti, An, Gt_re, "Gt_re");
	//for hybridization function
	FT(Diwn, wn, Dt, ti, B);	//save_data(ti, Dt, L, H, "Dt");
	AN(Dt, nth, An, ti, wi, "Dt");
	inverse(ti, An, Dt_re, "Dt_re");
	
	int extrema_index[32];	int peak_num = find_extrema(w, Dw, extrema_index);
	int num_extrema = 2*peak_num -1;	int half_index[2*peak_num];
	for(int i=0;i<peak_num;i++){
		FWHM(w, Dw, extrema_index, num_extrema, 2*i, half_index);
//		printf("%d,  %d\n", half_index[2*i], half_index[2*i+1]);
//		printf("extrema_index[%d] = %d\n",  2*i,  extrema_index[2*i]);
	}
	
	srand(time(NULL)+my_rank);

	int total_cycle = Repeat;
	int cycle_per_process = total_cycle/my_size;

	double* gather_chi = (double*)malloc( total_cycle * sizeof(double) );
	double* gather_bath = (double*)malloc( total_cycle * 4*Nb * sizeof(double) );

	double* chi = (double*)malloc( cycle_per_process * sizeof(double) );
	double* bath = (double*)malloc( cycle_per_process * 4*Nb * sizeof(double) );
	
	for(int i = 0; i < cycle_per_process; i++){
		init_bath( bath, i, extrema_index, half_index, peak_num );
		CGM( bath, chi, i );
	}
	MPI_Gather(chi, cycle_per_process , MPI_DOUBLE, gather_chi, cycle_per_process , MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(bath, cycle_per_process*4*Nb, MPI_DOUBLE, gather_bath, cycle_per_process*4*Nb, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	free(bath);	free(chi);

	// SORTING & SAVE 
	if( my_rank == 0 ) {
				
		int save_num = total_cycle > 10? 10: total_cycle;
		int* sortedIndices = (int*)malloc( total_cycle * sizeof(int));
		double* sorted_bath = (double*)malloc( save_num * 4*Nb * sizeof(double) );
		double* sorted_chi = (double*)malloc( save_num * sizeof(double) );
	
		for(int i = 0; i < total_cycle; i++)
			sortedIndices[i] = i;
		quickSort(gather_chi, sortedIndices, 0, total_cycle-1); 

		for(int i = 0; i < save_num; i++){
			sorted_chi[i] = gather_chi[i];
			for(int j = 0; j < 4*Nb; j++)
				sorted_bath[ i*4*Nb + j ] = gather_bath[ sortedIndices[i]*4*Nb + j ];	
		}
		
		
		// classify file -> naming -> save .h5 file 
		save_bath( sorted_bath, sorted_chi, save_num);
	
		free(sorted_bath);	free(sortedIndices);	
	}
	free(gather_bath);	free(gather_chi); 
}

void save_bath(double* sdata, double* chi, int snum){
	hid_t file_id, dataset_id, dataspace_id, group1_id, group2_id, attribute_id, attr_dataspace_id ;
	hsize_t dims[2], attr_dims[1];
	herr_t	status;
	double attr_data[0], bath[2][2*Nb];
	
	file_id = H5Fopen(FILE_h5, H5F_ACC_RDWR, H5P_DEFAULT);
	if(file_id < 0){
		file_id = H5Fcreate(FILE_h5, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	}	
	dims[0] = 2*Nb;	dims[1] = 2*Nb;

	char Group1[1024], Group2[1024], Data_num[128]; 
	sprintf(Group1, "/%s", H); sprintf(Group2, "%s%d", "N", Nb);
	
	group1_id = H5Gopen2(file_id, Group1, H5P_DEFAULT);
	if( group1_id < 0)
		group1_id = H5Gcreate2(file_id, Group1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	
	group2_id = H5Gopen2(group1_id, Group2, H5P_DEFAULT);	
	if( group2_id < 0)
		group2_id = H5Gcreate2(group1_id, Group2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
	
	for(int i = 0; i < snum; i++){
		 	
		for(int j = 0; j < 2*Nb; j++){
			bath[0][j] = sdata[i*4*Nb + j];
			bath[1][j] = sdata[i*4*Nb + 2*Nb + j];
		}
		
		sprintf(Data_num,"%c%d",'P',i);
		dataspace_id = H5Screate_simple(2, dims, NULL);
		dataset_id = H5Dcreate2(group2_id, Data_num, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bath);	
		
		attr_data[0] = chi[i];	attr_dims[0] = 1;	
		attr_dataspace_id = H5Screate_simple(1, attr_dims, NULL);	

		attribute_id = H5Acreate2(dataset_id, "chi", H5T_IEEE_F64BE, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
		status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, attr_data);
		
		status = H5Aclose(attribute_id);
		status = H5Sclose(attr_dataspace_id);
		status = H5Dclose(dataset_id);
	}
	status = H5Gclose(group2_id);
	H5Gclose(group1_id);
	H5Fclose(file_id);
}

// x: [-pi/a,pi/a], n,m,l:ratio between hopping terms , N: # of neighbor 
double E(double x, void* parmas){
//	a = 1
	struct E_params* p = (struct E_params*)parmas;
	double n,m,h;
	n = p->n; m = p->m; h = p->h; int N = p->N;
//	double e = 0;
	//normalization of bandwidth
	switch (N){
		case 1:
			return -.5*cos(x);	
		case 2:
			return -.5*( cos(x) + n*cos(2*x) ) / (1+n);
		case 3:
			return -.5*( cos(x) + n*cos(2*x) + m*cos(3*x) ) / (1+n+n*m);
		case 4:
			return -.5*( cos(x) + n*cos(2*x) + m*cos(3*x) + h*cos(4*x) ) / (1+n+n*m+n*m*h);
		default:
			printf("function E: error");
			return 0;
	}
}

double Gre(double x, void *p){
	struct G_params* params = (struct G_params*)p;
	struct E_params* e_params  = &(params->e_params);
	int n = (params->n);
	double complex W = *(params->w + n);
	
	return creal( 1./( W +  mu - E(x,e_params)) );
}

double Gim(double x, void *p){
	struct G_params* params = (struct G_params*)p;
	struct E_params* e_params  = &(params->e_params);
	int n = (params->n);
	double complex W = *(params->w + n);
	
	return cimag( 1. / ( W +  mu - E(x,e_params)) );
}

void calculate_GD(double *W, double complex *G,double complex *D,void*p,char A){ 
	struct G_params* params = (struct G_params*)p;
//	struct E_params* e_params = &(params->e_params);
	double gre, gim,abserr; abserr=0;
	
//	e_params->N = 1;
	
	if(A == 'r'){
		for(int n=0; n<Nmax; n++){
			W[n] = Wr(n,Nmax);
			params->w[n] = W[n]+(1.*I)*ETA;
		}
	}else{
		for(int n=0; n<Nmax; n++){
			W[n] = Wn(n,B);
			params->w[n] = (1.*I)*W[n];	
		}
	}
	
	gsl_integration_workspace *S = gsl_integration_workspace_alloc(N_EIGEN);
	gsl_function F1,F2;
	F1.function = &Gre;
	F1.params = params;
	F2.function = &Gim;
	F2.params = params;
	
	for(int n=0; n<Nmax; n++){
		gre = 0; gim =0;
		params->n = n;
					
		gsl_integration_qag(&F1,-PI,PI,epsabs,epsrel,N_EIGEN,6,S,&gre,&abserr);
		gsl_integration_qag(&F2,-PI,PI,epsabs,epsrel,N_EIGEN,6,S,&gim,&abserr);
		
		G[n] = (gre + (1.*I)*gim)/(2*PI);
		D[n] = params->w[n] + mu - 1./G[n];
	}
	
	gsl_integration_workspace_free(S);
	if(A=='r'){
		//save_data(W,G,Nmax,H,"Gw"); 
		//save_data(W,D,Nmax,H,"Dw");
	}else{
		//save_data(W,G,Nmax,H,"Giwn"); 
		//save_data(W,D,Nmax,H,"Diwn");
	}
}

void Gaussian(double a,double b, int len, double *ni, double *wi){
    //Gaussian Quadrature node & weights
    gsl_integration_glfixed_table *w;
    w = gsl_integration_glfixed_table_alloc(len);
    for(int i=0;i<len;i++){
        gsl_integration_glfixed_point(a,b,i,&ni[i],&wi[i],w);
//		prinf("w_%d = %f, x_%d = %f\n",i,wi[i],i,ni[i]);
    } 
    gsl_integration_glfixed_table_free(w);
}

double obj_f(const gsl_vector *v, void *p){
	double V[Nb],ep[Nb];
    double result = 0.;
	double complex D = 0.;
    
	for(int i=0;i<Nb;i++){
        V[i] = gsl_vector_get(v,i);     // 0 ~ Nb-1
        ep[i] = gsl_vector_get(v,Nb+i); // Nb ~ 2*Nb-1
    }
//	FILE* fp= fopen("ex","w");
	for(int n=0;n<lim;n++){
		D =0.;
		for(int j=0;j<Nb;j++){
			D += V[j]*V[j]/((1.*I)*wn[n]-ep[j]);
		}
        result += (creal( Diwn[n] -D ) * creal( Diwn[n] - D ) + cimag( Diwn[n] - D )*cimag( Diwn[n] - D ));//wn[n];
//		fprintf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\n",wn[n],creal(Diwn[n]),cimag(Diwn[n]),creal(D),cimag(D));	
	}
//	fclose(fp);
//	exit(0);
	return result/lim;
}

void obj_df(const gsl_vector* v, void* params, gsl_vector *df){
	double V[Nb],ep[Nb];
	for(int i=0;i<Nb;i++){
		V[i] = gsl_vector_get(v,i);
		ep[i] = gsl_vector_get(v,i+Nb);
//		printf("V[%d] = %f, e[%d] = %f\n",i,V[i],i,ep[i]);
	}
	double complex D;
	double dFdV,dFde;
	for(int i=0;i<Nb;i++){
		dFdV = 0.;	dFde = 0.;
		double sign = abs_sign(V[i]);
		for(int n=0;n<lim;n++){
			D=0.;
			for(int j=0;j<Nb;j++){
				D += cabs(V[j])*cabs(V[j])/((1.*I)*wn[n]-ep[j]);
			}
			dFdV += ( 4*cabs(V[i])*sign*(ep[i]*creal(Diwn[n]-D) + wn[n]*cimag(Diwn[n]-D)) / (wn[n]*wn[n]+ep[i]*ep[i]) );// wn[n];
			dFde += ( -2*V[i]*V[i]*( ( 2*ep[i]*ep[i]/(ep[i]*ep[i]+wn[n]*wn[n]) - 1.0 )*creal(Diwn[n]-D)
					+ 2*wn[n]*ep[i]*cimag(Diwn[n]-D)/(ep[i]*ep[i]+wn[n]*wn[n]) )/(ep[i]*ep[i]+wn[n]*wn[n]) ) ;// wn[n];
		}
//		printf("dFdV[%d] = %f , dFde[%d] = %f\n",i, dFdV/lim, i , dFde/lim);
		gsl_vector_set(df,i,dFdV/lim);
		gsl_vector_set(df,Nb+i,dFde/lim);
	}
/*	
	double dF=0.;     double f_plus,f_minus;
    double h = 1e-8;
    gsl_vector *v_plus = gsl_vector_alloc(v->size);
    gsl_vector *v_minus = gsl_vector_alloc(v->size);
    gsl_vector_memcpy(v_plus, v);
    gsl_vector_memcpy(v_minus, v);
	
	for(int i=0;i<Nb*2;i++){
		gsl_vector_set(v_plus,i,gsl_vector_get(v,i)+h);
		gsl_vector_set(v_minus,i,gsl_vector_get(v,i)-h);
        f_plus = obj_f(v_plus, params);
        f_minus = obj_f(v_minus, params);
        dF = ( f_plus - f_minus ) / (2.*h);
		printf("dF[%d] = %f\n",i, dF);
		gsl_vector_set(v_plus,i,gsl_vector_get(v,i));
		gsl_vector_set(v_minus,i,gsl_vector_get(v,i));
	}
    // Clean up
    gsl_vector_free(v_plus);
    gsl_vector_free(v_minus);
*/
}

void obj_fdf(const gsl_vector *x, void *params, double *f, gsl_vector *df){
    *f = obj_f(x, params);
    obj_df(x,params,df);
}

double derivative(double *x, double complex *y,int index, int size){
	double h = (x[size-1] - x[0])/(size-1);
	double dy;
	if(index>0 && index<size-1){
		dy = cimag(y[index+1] - y[index-1])/(2.*h);
	}else if (index==0){
		dy = cimag(y[1] -y[0])/(2.*h);
	}else if (index==size-1){
		dy = cimag(y[size-1] - y[size-2])/h;
	}else {
		return NAN;
	}
	return dy;
}

int find_extrema(double *x, double complex *y, int* extrema_index){
	int peak_num = 0;
	int iter = 0;
	int c[100];	memset(c,0,sizeof(c));	//save epsilon
	double prev_dy = derivative(x,y,0,Nmax);
	
	for(int i=1;i<Nmax;i++){
		double dy = derivative(x,y,i,Nmax);
		if( (prev_dy < 0 &&  dy > 0) || (prev_dy > 0 && dy <0) ){
//			printf("w[%d]=%f: prev_dy = %f, dy = %f\n",i,w[i],prev_dy,dy);
			c[iter] = ( (i-1) + i ) / 2;
			iter++;
			if(prev_dy<0 && dy>0){
				peak_num++;
			}
		}
		prev_dy = dy;
	}
	for(int i=0;i<iter;i++){
		extrema_index[i] = c[i];
//		printf("extrema_index[%d] = %d\n",i,extrema_index[i]);
	}
//	printf("peak_num = %d\n",peak_num);	
	return peak_num;
}	

double generate_randnum(double num,double err){
	return num*(1.-err)+ ( (double)rand() / RAND_MAX ) * num*(2*err);
}

double randnum(double min,double max){
	return min + ( (double)rand() / RAND_MAX ) * (max - min);
}

void generate_V(double* bath, int iter, double err_range){
    double v[Nb];
    double sumOfSquares = 0.0;
    double const_sum = -integral_imag(w,Dw,Nmax)/PI/Nb;
	double MIN_RANGE_PERCENT = 1. - err_range;
	double MAX_RANGE_PERCENT = 1. + err_range;
	// Generate random values for v_0 to v_(Nb-2) within the specified range
	for (int i = 0; i < Nb - 1; i++) {
		double minValue = (MIN_RANGE_PERCENT * sqrt(const_sum));
		double maxValue = (MAX_RANGE_PERCENT * sqrt(const_sum));
        v[i] = randnum(minValue,maxValue);
        sumOfSquares += v[i] * v[i];
    }
	
    // Calculate the square of v_10 to maintain the desired sum of squares
    double lastValueSquared = fabs(const_sum*Nb - sumOfSquares);
	
    v[Nb- 1] = sqrt(lastValueSquared);
	
    // Display the values
//	double sum = 0.;
	for (int i = 0; i < Nb; i++) {
        bath[iter*4*Nb + i] = v[i];
//		printf("%.2f ", bath[iter*4*Nb + i]);	
//		sum += v[i]*v[i];
	}
//	printf("\n");
//	printf("const_sum = %f, sum of init_V^2 = %f\n",const_sum*Nb,sum);
}

void randSplit(int N, int n, int* splits){
	for(int i=0;i<n;i++)
		splits[i] = 0;
	
	for(int i=0;i<n-1;i++){
		splits[i] = rand() % (N-1);
		N -= splits[i];
	}
	splits[n-1] = N;
}

void ArrToStr(int* arr, int n, char* str, int size, int used){
	int offset= used;
	for(int i=0;i<n-1;i++){
		offset += snprintf(str + offset, size-offset,"%d,",arr[i]);
//		printf("offset = %d\n",offset);
	}
	offset += snprintf(str+offset,size-offset,"%d",arr[n-1]);
	offset += snprintf(str+offset,size-offset,"_%ld",time(NULL));
}

void generate_ep(double* bath, int iter, int* ex_index, int* half_index, int peak_num ){
	int splits[peak_num];
	int N = Nb-3;
	randSplit(N,peak_num,splits);
	for(int i = 0; i < peak_num; i++){
		bath[iter*4*Nb + N+3]= w[ex_index[2*i]];
		for(int j = 1; j <= splits[i]; j++){
			bath[iter*4*Nb + N+3+j] = randnum(w[half_index[2*i]],w[half_index[2*i+1]]);
		}
		splits[i]++;
		N += splits[i];
	}
//	for(int i = 0; i < Nb; i++)
//		printf("%.2f " , bath[iter*4*Nb + Nb + i]);
//	printf("\n");
/*	int under0 = 0;
	for(int i=0;i<Nb;i++){
		if( bath[iter][Nb+i] < 0 ){
			under0 ++;
		}
	}
	int offset = snprintf(fname,1024,"%d,%d_",under0,Nb-under0);	
	ArrToStr(splits,peak_num,fname,1024,offset);
*/
}

void init_bath( double* bath, int iteration, int* ex_index, int* half_index, int peak_num ){
	generate_V(bath, iteration, 0.);
	generate_ep(bath, iteration, ex_index, half_index, peak_num);
	
//	sprintf(fname, "6, 13_3, 9, 7_1696475463_1_1.txt"); call_bath(fname, bath);

/*	char path[1024];	sprintf(path, "%s/%d/%s",  H, Nb, "Rhyb_i");
	hyb(Rhyb_i, bath, 'r');	save_hyb(w, Rhyb_i, Nmax, Nb, path, fname);
	char path2[1024];	sprintf(path2, "%s/%d/%s", H, Nb, "Ihyb_i");
	hyb(Ihyb_i, bath, 'i');	save_hyb(wn, Ihyb_i, Nmax, Nb, path2, fname);
	char bathpath[1024];	sprintf(bathpath, "%s/%d/%s", H, Nb, "bath_i");
	save_bath(bath, Nb, bathpath,fname);
*/
}

void call_bath(char* fname, double* bath){
    char fpath[1024];   sprintf(fpath,"%s/%d/%s/%s", H, Nb, "bath_f", fname);
    FILE* file = fopen(fpath, "r");
    if(file == NULL)
        perror("Can't open file");
    for(int i=0;i<Nb;i++){
        if(fscanf(file, "%lf\t%lf", &bath[i+Nb], &bath[i]) != 2 ){
            break;
        }   
    }   
    fclose(file);
    char newfn[256]; // Make sure it's large enough to hold the result
    char* originalName = fname; // Dereference the pointer to get the original file name
    char* dotp = strrchr(originalName, '.');
    if (dotp != NULL) {
        size_t prefixLength = dotp - originalName;
        strncpy(newfn, originalName, prefixLength);
        newfn[prefixLength] = '\0';
        strncat(newfn, "_1", sizeof(newfn) - strlen(newfn) - 1); 
        strcpy(originalName, newfn); // Copy the modified name back to the original pointer
		printf("call: %s\n", fname);
    } else {
        printf("Error: The original file name does not contain a dot.\n");
    }   
}

//peak_index: 극점의오름차순 index, peak_num: 극점 개수, peak: 계산하려는 폭의 중심이 되는 극소점의 index
void FWHM(double* x, double complex* y, int* peak_index, int peak_num, int peak, int* half_index){ //Find Half Maximum
	double half;
	if(peak_num != 1 ){
		if(peak != peak_num-1){
			half = cimag(y[peak_index[peak]]) - ( cimag(y[peak_index[peak]]) - cimag(y[peak_index[peak+1]]) ) * 0.65;
		}else if(peak == peak_num-1){
			half = cimag(y[peak_index[peak]]) - ( cimag(y[peak_index[peak]]) - cimag(y[peak_index[peak-1]]) ) * 0.65;
		}
	}else{
		half = cimag(y[peak_index[peak]])/2;
	}

	int left_index = peak_index[peak]-2;
	while(left_index > 0 && cimag(y[left_index]) < half){
		left_index--;
	}
	half_index[peak] = left_index;
	
	int right_index = peak_index[peak]+2;
	while(right_index <Nmax-1 && cimag(y[right_index]) < half ){
		right_index++;
	}	
	half_index[peak+1] = right_index;
//	printf("left=%d, right=%d\n",left_index, right_index);	
//	return x[right_index]-x[left_index];
}

void CGM( double* bath, double* chi, int iteration ){
	double f, prev_f;
	size_t iter;	size_t iter2=0;
    int status;
	double tol = 1e-3;
	
	const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;
	gsl_vector *x;
    gsl_multimin_function_fdf obj_func;
	
	obj_func.n = 2*Nb;
    obj_func.f = obj_f;
    obj_func.df = obj_df;
    obj_func.fdf = obj_fdf;
    obj_func.params = NULL;
	
	// sets the value of the i-th element of a vector v to x.
    x = gsl_vector_alloc(2*Nb);
	for(int i=0;i<2*Nb;i++){
        gsl_vector_set(x, i, bath[iteration*4*Nb + i]);		
	}
	T = gsl_multimin_fdfminimizer_vector_bfgs2;
	s =  gsl_multimin_fdfminimizer_alloc (T, 2*Nb);
	
	do{
		gsl_multimin_fdfminimizer_set(s, &obj_func, x, stepsize, tol);	
	
		iter = 0;
		prev_f	= gsl_multimin_fdfminimizer_minimum(s);
		do{
			iter++;
			status = gsl_multimin_fdfminimizer_iterate (s);
			if (status == GSL_SUCCESS) {
//				printf("최적화가 성공적으로 완료되었습니다.\n");
			} else if (status == GSL_CONTINUE) {
//				printf("최적화가 진행 중입니다.\n");
			} else if (status == GSL_ENOPROG) {
//				printf("진전이 없어 최적화를 종료합니다.\n");
			} else {
				printf("최적화 중 오류가 발생하여 종료합니다.\n");
			}
			if (status)
				break;
			status = gsl_multimin_test_gradient(s->gradient, 1e-8);
		}while (status == GSL_CONTINUE && iter<10000);

		for(int i=0;i<2*Nb;i++)	
			gsl_vector_set(x,i, gsl_vector_get(s->x, i));
		
		f = gsl_multimin_fdfminimizer_minimum(s);//	printf("f = %e\n", f);	
//		gsl_multimin_fdfminimizer_restart(s);
//		iter2++;printf("iter2: %ld , %.15e \n", iter2, fabs(f - prev_f));
		
	}while (fabs(f - prev_f) > 1e-16 && iter2<1000);
	
	f = gsl_multimin_fdfminimizer_minimum(s);	
	for(int i=0; i<2*Nb; i++){
		bath[iteration*4*Nb + 2*Nb + i] = gsl_vector_get(s->x, i);
//		printf("%.2f ", bath[iteration*4*Nb + 2*Nb +i] );
	}
//	printf("\n");
	chi[iteration] = f;	
	
/*	
	char path[1024];	sprintf(path, "%s/%d/%s", H, Nb, "bath_f");
	//save_bath(bath_f, Nb, path, fname);
	//save_add(path, fname, "final_f", &f, 1);		
	char path2[1024];	sprintf(path2,"%s/%d/%s", H, Nb, "Rhyb_f");
	hyb(Rhyb_f, bath_f, 'r'); //save_hyb(w, Rhyb_f, Nmax, Nb, path2, fname);
	char path3[1024];	sprintf(path3, "%s/%d/%s", H, Nb, "Ihyb_f");
	hyb(Ihyb_f, bath_f, 'i'); //save_hyb(wn, Ihyb_f, Nmax, Nb, path3, fname);	
	
	printf("CGM: filename = %s ->  f() = %.10e\n", fname, s->f);
*/	
	gsl_multimin_fdfminimizer_free (s);
    gsl_vector_free (x);
}

void hyb(double complex *D, double *bath, char domain){
	if( domain == 'r' ){
		for(int n=0;n<Nmax;n++){
			D[n] = 0.;
			for(int i=0;i<Nb;i++){
				D[n] += bath[i]*bath[i]/(w[n] + (1.*I) * ETA-bath[i+Nb]);
			}
		}
	}else{
		for(int n=0;n<Nmax;n++){
			D[n] = 0.;
			for(int i=0;i<Nb;i++){
				D[n] += bath[i]*bath[i]/((1.*I)*wn[n]-bath[i+Nb]);
			}
		}
	}
}

double bisection(double desired, double* arr_x, double* arr_y, int size){
	int left = 0;
	int right = size - 1;
	while( left<=right){
		int mid = (left+right)/2;
		double mid_value = (arr_y[left]+arr_y[right])/2.0;
		if( fabs(mid_value-desired) < 1e-6 ){
			return arr_x[mid];
		} else if ( mid_value-desired > 1e-6){
			right = mid - 1;
		} else {
			left = mid + 1;
		}
	}
	double closest_left = arr_x[right];
	double closest_right = arr_x[left];
	return (closest_left + closest_right)/2.0;
}

void fermiNdos(void *params){
	mu = 0;
	struct G_params* p = (struct G_params*)params;
//	struct E_params* e_params = &(p->e_params);

	double DOS[N_DOS];	  memset(DOS, 0, sizeof(DOS));
	double energy[N_EIGEN];
	double sum_DOS[N_DOS-1]; memset(sum_DOS, 0, sizeof(sum_DOS)); // # of states(electorns)
	double dos, abserr;
	
	for(int n=0; n<N_EIGEN; n++)
		energy[n] = Wr(n, N_EIGEN);
			
	gsl_integration_workspace *S = gsl_integration_workspace_alloc(N_DOS);
	gsl_function F;
	F.function = &Gim;
	F.params = p; 
	for(int n=0;n<N_DOS;n++){
		dos = 0; abserr=0;
		p->n = n;
		p->w[n] = energy[n] + (1.*I)*ETA;
		gsl_integration_qag(&F, -PI, PI, epsabs, epsrel, N_EIGEN, 6, S, &dos, &abserr);
		DOS[n] = -dos/(2*PI)/PI; // (1/2PI): because k sum & (-1/PI): kk relation	
	}
	gsl_integration_workspace_free(S);
	//save(energy, DOS, N_DOS, H, "DOS");
	
	double dE = (energy[N_EIGEN-1] - energy[0])/(N_EIGEN-1);
	sum_DOS[0] = (DOS[1]+DOS[0])/2.*dE;	
	for(int i=0;i<N_DOS-1-1;i++){
		sum_DOS[i+1] = sum_DOS[i] + (DOS[i+2]+DOS[i+1])/2.*dE;
	}
	//save(energy, sum_DOS, N_DOS-1, H, "sum_DOS");
	
	mu = bisection(0.5, energy, sum_DOS, N_DOS-1);
//	printf("Fermi Value = %f\n", mu);
}

/*void Error(double *x, double complex *f, double complex *g, double complex* h, int size, char* fname){
	double complex Err_i[size], Err_f[size];
	for(int i=0;i<size;i++){
		Err_i[i] = (f[i] - g[i]);
		Err_f[i] = (f[i] - h[i]);
	}
	char path1[1024];	sprintf(path1, "%s/%d/%s", H, Nb, "Err_i");
	//save_err(x, Err_i, size, path1, fname);
	
	char path2[1024];	sprintf(path2, "%s/%d/%s", H, Nb, "Err_f");
	//save_err(x, Err_f, size, path2, fname);
}*/

void AN(double complex *func, double* x, double complex* A, double *ti, double *wi, char* fname){
// calculate Legendre coeff using Gaussian quad
    double (*p)(int, double);
    p = P;

    for(int i=0; i<Amax; i++){
        double complex c=0;
        x[i] = (double)i;
        for(int j=0; j<L; j++){
            c += func[j]*p(i, 2*ti[j]/B-1)*wi[j];
        }
        A[i] = c*(2*i+1)/B;
    }
	char path[50]; sprintf(path, "%s/%s", H, "An");
	//save_An(x, A, Amax, path, fname);
}

void LT( double *ti, double complex *A, double complex *Y, int num){
// Legendre transformation
    double (*p)(int, double);
    p = P;
	double complex f;
    for(int i=0;i<L; i++){
        f = 0;
        for(int j=0; j<num; j++){
            f += A[j]*p(j, 2*ti[i]/B-1);
        }
        Y[i] = f;
    }
}

void FT(double complex* b, double* node1, double complex* af, double* node2, double cycle){
// Fourier transformation using Gaussian quad
	int b_point_num = Nmax;
    int af_point_num = L;
    double complex f =0;
    for(int i=0; i<af_point_num; i++){  //tau index
        f = 0.;
        for(int j=0; j<b_point_num; j++){   // Wn index
            f += (cos(node1[j]*node2[i]) - (1.*I) * sin(node1[j]*node2[i])) * (b[j]-1 / ( (1.*I) * node1[j] ));
        }
        af[i] = f/cycle-0.5;
    }
}

void inverse( double* ti, double complex* A, double complex* y, char* folder){
    int num[5]; for(int i=0;i<3;i++){ num[i]=3+2*i; }; num[3]=30; num[4] = Amax;
	char path[50]="";	snprintf(path, sizeof(path), "%s/%s", H, folder);
    for(int i=0;i<5;i++){
        char fname[1024];	sprintf(fname, "%d", num[i]);
        LT(ti, A, y, num[i]);
        //save_re(ti, y, L, path, fname);
    }
}

void pathname(void* params){
	struct G_params* p = (struct G_params*)params;
	int N = p->e_params.N;
	if( N == 1){
		sprintf(H, "N_%1d", N);
	}else if ( N == 2 ){
		sprintf(H, "N_%1dn_%02d", N, (int)floor(0.5+p->e_params.n*100));
	}else if ( N == 3){
		sprintf(H, "N_%1dn_%02dm_%02d", N, (int)floor(0.5+p->e_params.n*100), (int)floor(0.5+p->e_params.m*100));
	}else if ( N == 4){
		sprintf(H, "N_%1dn_%02dm_%02dh_%02d", N, (int)floor(0.5+p->e_params.n*100), (int)floor(0.5+p->e_params.m*100), (int)floor(0.5+p->e_params.h*100));
	}
}

double integral_imag(double *x, double complex *y, int size){
	double result = 0.;
	double dx = (x[size-1] - x[0]) / (size-1);
	for(int i = 1; i < size; i++){
		result += ( cimag(y[i-1]) + cimag(y[i]) ) * dx /2.;
	}
	return result;
}

double abs_sign(double x){
	if(x<0){
		return -1.;
	}
	else{
		return 1.;
	}
}

// Swap two elements in an array
void swap_int(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void swap_double(double* a, double* b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

// Partition the array into two sub-arrays and return the pivot index
int partition(double* arr, int* indices, int low, int high) {
    double pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap_double(&arr[i], &arr[j]);
			swap_int(&indices[i], &indices[j]);
        }
    }

    swap_double(&arr[i + 1], &arr[high]);
	swap_int(&indices[i + 1], &indices[high]);
    return (i + 1);
}

// Main quick sort function
void quickSort(double* arr, int* indices, int low, int high) {
    if (low < high) {
        int pi = partition(arr, indices, low, high);
        quickSort(arr, indices, low, pi - 1);
        quickSort(arr, indices, pi + 1, high);
    }
}
