//*************************************************************************************
//******* 2D POISSON's EQUATION USING MPI PARALLELISATION****************************//
//*************************************************************************************
/*
    In this Program uptill now , we have robust domain decomposition which divides full domain into subdomain
    in 1D linear fashion in y-direction ,Then cartesian toppology is set-up for easy communication between  processor .
    Then residual array is calculated by rank 0 processor and then it is distributed among given number of processor .
    Memory allocation takes places according to their position in cartesian topology . 

    Now ... cg algorithm works only on r ..so every procssor will iterate with r and whenever u has to be calculated,
    it will be calculated by rank 0 for easy visualization . 

*/

//***************************
//*** INCLUDES ************//
//***************************

#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <fstream>

//***************************************************
//****** GLOBAL VARAIBLES *************************//
//*************************************************** 

const double PI = 3.14;
const double con_sinh = 267; // Has to be changed afterwards
const double tol = 0.0001;

//**************************************************
//******* FUNCTION DECLARATIONS *******************//
//**************************************************

inline void domaindecompose(const int &numx,const int &numy,const int &size,int domain[]);
inline void Boundary_Init(const int &numx,const int &numy,const double hx, double *u);
inline void func_init(const int &numx,const int &numy,const double &hx,const double &hy,std::vector<double>& f);
inline double Compute_Matrix_Vec_Mult(const double &u_center,const double &u_left,const double &u_right,const double &u_up,const double &u_down,const double &constant,const double &hxsqinv,const double &hysqinv);
void calInnerGridIndexes(const int &numGridX,const int &numGridY,std::vector<int>& ind) ;
inline void write_sol(const std::vector<double>& u_temp,const double hx,const double hy,const int numx);
inline bool terminate(const double value);
inline double Scalar_Product(const int temp_r[],const int size);
//void CG_Paralle_Solver(std::vector<double>& u, const std::vector<double>& f, std::vector<double>& r,const std::vector<int>& ind,const double hxsqinv,const double hysqinv,const double constant,const int num_iter,const int numx);
void transmit_r_u_local(double *r, double *u, const int *domain, double *local_u, double * local_d,double * local_r,const int numgridpoints_x, const int cart_rank,const int size,MPI_Comm new_comm);
void cg_Init(std::vector<double> f, double *r,double *u,const int &numgridpoints_x, const int &numgridpoints_y, const double &constant, const double &hxsqinv, const double &hysqinv);
   


//*****************
//** MAIN()******//
//*****************
int main(int argc, char *argv[])
{
	int size(0);
	int rank(0);

	// *************************************
	// *** MPI Initialisation **************
	//**************************************
	MPI_Init(&argc,&argv);
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  	// Checking whether number of parameters passed during main call are as per the requirement or not and if 
  	// they are not , program is terminated and this is checked by processor 0 

  	if (rank==0)
	{
  		if (argc!=4) {
   		std::cout << "Aborting the Program , Please check the number of the input arguments" << std::endl;
    	exit(1);
  	}
  		else
    	std::cout << "OK .....Parameters Taken" << std::endl;
	}

	//********************************************************************
	//****** PARAMETER INITIALISATION **********************************//
	//********************************************************************
	
	  std::string s1 = argv[1],s2= argv[2],s3 = argv[3];
  	int numx_interval = std::stoi(s1) ;
  	int numy_interval = std::stoi(s2);
  	int num_iter = std::stoi(s3);
  	int numgridpoints_x = numx_interval+1;
  	int numgridpoints_y = numy_interval+1;
  	const double hx  = 2.0 / (numx_interval);
  	const double hy = 1.0 / (numy_interval);
  	const double hxsq = hx * hx ;
  	const double hysq = hy * hy ;
  	const double hxsqinv = 1.0/ hxsq;
  	const double hysqinv = 1.0/ hysq;
  	const double K = 2 * PI;
  	const double constant = (2*hxsqinv) + (2*hysqinv) + (K*K);
  	double local_residual= 0.0;
  	int domain[5] ={0},sizeproc;
    int numTotGridPoints = numgridpoints_x*numgridpoints_y;
    double r[numTotGridPoints]={0.};
    double u[numTotGridPoints]= {0.};
    double d[numTotGridPoints]={0.};


  	//***************************************
  	//** Domain Decompose ***************** //
  	//***************************************
  	if (rank==0)
  	{
		  domaindecompose(numgridpoints_x,numgridpoints_y,size,domain);
      sizeproc = domain[1]*numgridpoints_x;
      MPI_Bcast(&sizeproc, 1, MPI_INT, 0, 
               MPI_COMM_WORLD);

  	}

  //  int sizeproc = domain[1]*numgridpoints_x;
     std::cout << "The data points assigned to processor is: " << sizeproc << std::endl;
     
  	//***************************************
  	//******Cartesian Cordinate System ****** 
  	//***************************************

    MPI_Comm new_comm(MPI_COMM_NULL);
    int ndims = 2;
    int dim[ndims] = {1,size};
    int periods[ndims] = {0};
    int reorder = 1;
    int up,down = 0;
    int coords[2] ={0,0};
    int cart_rank(0);
    
    //std::cout << "My ranks before topology construction is :" << cart_rank << "\n";
    MPI_Cart_create(MPI_COMM_WORLD,ndims,dim,periods,reorder,&new_comm);
    MPI_Comm_rank(new_comm,&cart_rank);
    //std::cout << "My cart rank is :" << cart_rank << "\n";
    MPI_Cart_coords(new_comm,cart_rank,ndims,coords);
    // std::cout << "My cartesian coordinates is : (" << coords[0] << ","  << coords[1] << ") \n";
    MPI_Cart_shift(new_comm,1,1,&down,&up);
    //std::cout<<"this is my rank: "<<cart_rank<<std::endl;
    //std::cout<<"This is my neighbours: "<<up<<"  "<<down<<std::endl;

    // Allocating local space for each processor
    // double local_r[sizeproc]={0.};
    // double local_u[sizeproc]={0.};
    // double local_d[sizeproc]={0.};
    // double ghost_d[2*numgridpoints_x]={0.};
    

    //***********************************************
    //**** Initialize boundary conditions of u(x,y) and f(x,y) values***//
    //***********************************************
    if(cart_rank==0)
    {
    
    //**********************************************************************
    //Boundary Init for u
    //**********************************************************************  
    //Boundary_Init(numgridpoints_x,numgridpoints_y,hx,u);

    //**********************************************************************
    //Functionc Init
    //**********************************************************************
    std::vector<double> f(numgridpoints_x*numgridpoints_y,0.0);
    func_init(numgridpoints_x, numgridpoints_y,hx,hy,f);

    cg_Init(f,r,u,numgridpoints_x,numgridpoints_y,constant,hxsqinv,hysqinv);
    
    } 

    transmit_r_u_local(r,u,domain,local_u,local_d,local_r, numgridpoints_x,cart_rank, size, new_comm);

  	//********************************************
  	// Allocating memory in each processor *****//
  	//********************************************
  	// Memory in each processor allocated according to domain array 
    
    // Communicate 
    
   //  if(cart_rank == 0)
   //  {
   //    if (domain[2]==0)
   //    {
   //      sizeproc = domain[1] * numgridpoints_x;
   //      for (int i = 1; i < size; ++i)
   //      {
   //        MPI_Send(&sizeproc,1,MPI_INT,i,0,new_comm);
   //      }
   //    }
   //    else
   //    {
   //      sizeproc= domain[1] * numgridpoints_x;
   //      int sizeproc_2= domain[3] * numgridpoints_x;
   //      for (int i = 1; i < size-1; ++i)
   //      {
   //        MPI_Send(&sizeproc,1,MPI_INT,i,0,new_comm);
   //      }
   //        MPI_Send(&sizeproc_2,1,MPI_INT,size-1,0,new_comm);
   //    }
   //  }
	  // else
	  // {
  	// 	MPI_Status status;
  	// 	MPI_Recv(&sizeproc,1,MPI_INT,0,0,new_comm,&status);
  	// 	//std::cout<<"I am Rank "<<rank<<" and I have received "<<sizeproc<<"memeory for computation"<<std::endl;
  	// }

    // Memory allocation for each processor according to cartesian topology
   // if(down<0 || up <0)
   //  {
   //    //std::cout<<"I am rank "<<cart_rank<<std::endl;
   //    //std::cout<<"I am allocating memeory    "<<sizeproc+numgridpoints_x<<" for computation"<<std::endl;
     
   //    //Allocating for r
   //    double local_r[sizeproc]={0};
   //    //Allocating for d
   //    double local_d[sizeproc+(1*numgridpoints_x)]={0};
    

   //  }
   //  else
   //  {
   //    //std::cout<<"I am rank "<<cart_rank<<std::endl;
      //std::cout<<"I am allocating memory    "<<sizeproc+(2*numgridpoints_x)<<"for computation"<<std::endl;
     
      //Allocating for r
      
    
     
    //******************************************************
    //*** Distribution of r into several processors ********
    //****************************************************** 

    //***************************************************************
  
  MPI_Finalize();

return 0;
}
//****************************************
// Domain decomosition 
//****************************************
inline void domaindecompose(const int &numx,const int &numy,const int &size,int domain[])
{
	std::cout<<"Domain Decmposition starts for 1D array of processor"<<std::endl;	
	int dataprocydirect = 0;
	int numprocxdirect = 1;
	if(numy%size ==0)
	{
		dataprocydirect = numy / size;
		domain[0] = size;
		domain[1] = dataprocydirect;
		domain[2] = 0;
		domain[3] = 0;
    domain[4] = 1;
		std::cout<<"Domain is Decomposed into equal subdomains "<<std::endl;
	}
  else
  {
    if(numy%(size-1)!=0)
    {
      dataprocydirect = numy / (size-1);
      domain[0] = size-1;
      domain[1] = dataprocydirect;
      domain[2] = 1;
      domain[3] = numy- (domain[0] * domain[1]);
      domain[4] = 2;
      std::cout<<"Domain is  Decomposed into irregular domains 2nd form "<<std::endl;
    }
    else
    {
      dataprocydirect = numy / size;
      domain[0] = size-1;
      domain[1] = dataprocydirect;
      domain[2] = 1;
      domain[3] = numy - (domain[0]*domain[1]);
      domain[4] = 3;
      std::cout<<"Domain is Decomposed into irregular subdomains 3rd form"<<std::endl;
    }

  }
  
	
}
//******************************
// Boundary value Initialisation 

inline void Boundary_Init(const int &numx,const int &numy,const double &hx, double *u){
  // Implementig the boundary value
  for (int i = (numy-1) * numx; i < (numx * numy) ; ++i ) {
      u[i] = (con_sinh* sin(2 * PI * (hx *(i - ((numy - 1) * numx)))));
      //std::cout<<"The value of u is"<<u_temp[i]<<std::endl;
  }
  //std::cout << "Checking boundary value initilization" << std::endl;
}
//******************************

//******************************
// Functional Initialisation 

inline void func_init(const int &numx,const int &numy,const double &hx,const double &hy,std::vector<double>& f){
  // Updating the function value on any domain
  double x,y;
  for (int  i = 0; i < numx * numy; ++i) {
    x = hx * (i % numx);
    y = hy * (i / numx);
    f[i] = 4 * PI *PI * sin(2 * PI * x) * sinh(2* PI * y );
    //std::cout << "The value of f is"<<f[i]<< std::endl;

  }
}
//******************************

//******************************
// Compute Matrix vector Multiplication 

inline double Compute_Matrix_Vec_Mult(const double &u_center,const double &u_left,const double &u_right,const double &u_up,const double &u_down,const double &constant,const double &hxsqinv,const double &hysqinv){

  return(((u_center * (constant) - (((hxsqinv)*(u_left + u_right)) + (hysqinv)*( u_up + u_down)))));
}

//******************************
// Write solution to file 
inline void write_sol(const std::vector<double>& u_temp,const double hx,const double hy,const int numx){

  double x,y;
  std::ofstream file_;
  file_.open("solution.txt");
  for (size_t i = 0; i < u_temp.size(); i++) {

    x = hx * (i % numx);
    y = hy * (i / numx);
    file_<<x<<"  "<<y<<"  "<<u_temp[i]<<std::endl;
  }
  file_.close();
  std::cout << "The solution has been written in solution.txt" << std::endl;

}

//*******************************
// Terminating function 
inline bool terminate(const double value){
  if(value<tol){
    std::cout << "The solution is converged...exiting" << std::endl;
    std::cout << "The residual is " <<value<< std::endl;
    //exit(0);
    return true;
  }
  else
    return false;
}

//*****************************
// Scalar product function 
inline double Scalar_Product(const double r[],const int size){
  // Calculating the norm of the vector
  double sum(0);
  for (size_t i = 0; i < size; i++) {
    // Computing the Norm
    sum+= r[i]*r[i];

  }
  return (sum);
}

//*****************************************************
/*void CG_Paralle_Solver(std::vector<double>& u, const std::vector<double>& f, std::vector<double>& r,const std::vector<int>& ind,const double hxsqinv,const double hysqinv,const double constant,const int num_iter,const int numx) {

  // Computing the Residual
  for (size_t i = 0; i < ind.size(); ++i) {
    double u_left= u[ind[i]-1];
    double u_right= u[ind[i]+1];
    double u_up= u[ind[i]+numx];
    double u_down = u[ind[i]-numx];
    double u_center= u[ind[i]];
    r[ind[i]]= f[ind[i]] - Compute_Matrix_Vec_Mult(u_center,u_left,u_right,u_up,u_down,constant,hxsqinv,hysqinv);
  }

  // Computing the norm
  double delta_r=Scalar_Product(r);  // include sqrt in the Scalar_Product and change the name of this function
  //std::cout << "The norm of the function is" <<delta_r<< std::endl;

  // Check for terminating conditions
  if(terminate(sqrt(delta_r))) return;

  //allocate d=r
  std::vector<double> d=r;

  //create d
  std::vector<double> z(d.size(),0);

  for (int j = 0; j < num_iter; j++) {   // replace with while loop

    // z= Ad;
    for (size_t i = 0; i < ind.size(); i++) {
      double d_left= d[ind[i]-1];
      double d_right= d[ind[i]+1];
      double d_up= d[ind[i]+numx];
      double d_down = d[ind[i]-numx];
      double d_center= d[ind[i]];
      z[ind[i]]= Compute_Matrix_Vec_Mult(d_center,d_left,d_right,d_up,d_down,constant,hxsqinv,hysqinv);
    }

    // Compute Scalar Product
    double sum =0;  // declare sum outside for loop
        for (size_t i = 0; i < d.size(); i++) {
      sum+= z[i]*d[i];
    }

    // Computing Alpha
    double suminv= 1.0/sum;      // no need to define suminv
    double alpha= suminv* delta_r;

    // u+ alpha*d

    for (size_t i = 0; i < u.size(); i++) {
      u[i]+= alpha*d[i];
    }

    // r= r- alpha*z

    for (size_t i = 0; i < r.size(); i++) {
      r[i]-= alpha*z[i];
    }

    double delta_r_1= Scalar_Product(r);
    if(terminate(sqrt(delta_r_1)))
      break;

    // beta = delta_r_1/delta_r;
    double delta_r_inv = 1/delta_r;    // no need to define delta_r_inv and all the variables to be defined outside for loop
    double beta= delta_r_1 * delta_r_inv;

    for (size_t i = 0; i < d.size(); i++) {
      d[i]= r[i] + beta*d[i];
    }

    delta_r=delta_r_1;

  }

}*/

// This function computes the intial steps of cg method
    void cg_Init(std::vector<double> f, double *r,double *u,const int &numgridpoints_x, const int &numgridpoints_y, const double &constant, const double &hxsqinv, const double &hysqinv)
    {
      int numTotGridPoints = numgridpoints_x*numgridpoints_y;
      //**********************************************************************
      //Calculate inner grid indexes
      //**********************************************************************
        std::vector<int> ind;
        calInnerGridIndexes(numgridpoints_x,numgridpoints_y,ind);
       // for (size_t i = 0; i < ind.size(); i++) 
       // std::cout << "The corect indices are " <<ind[i]<< std::endl;

        // r = f - Au
      for (size_t i = 0; i < ind.size(); i++) {
        double u_left= u[ind[i]-1];
        double u_right= u[ind[i]+1];
        double u_up= u[ind[i]+numgridpoints_x];
        double u_down = u[ind[i]-numgridpoints_x];
        double u_center= u[ind[i]];
        r[ind[i]]= f[ind[i]] - Compute_Matrix_Vec_Mult(u_center,u_left,u_right,u_up,u_down,constant,hxsqinv,hysqinv);
      }

      double delta_0 = Scalar_Product(r,numTotGridPoints);
      if(terminate(sqrt(delta_0)) ) return;
      //int stop = 1// stop;

      // d = r step would be done in next function  
    }

 // In this function processor 0 transmits the global u and r array to the each processors
    void transmit_r_u_local(double *r, double *u, const int *domain, double *local_u, double * local_d,double * local_r,const int numgridpoints_x, const int cart_rank,const int size,MPI_Comm new_comm)
    {

    // Distribution of r if domain is decomposed in first form ******
      int sizeproc,sizeproc_2;
    if(cart_rank==0)
    {
      sizeproc = domain[1]*numgridpoints_x;
      if(domain[4]==1)
      {
        for (int i = 1; i < size; ++i)
        {
          /* Sending r  */
          MPI_Send((r+ (i* sizeproc)),sizeproc,MPI_DOUBLE,i,0,new_comm);

        }

        /*sending u for the last process as for others its zero
        send u values  with tag 1 to avoid confusion
        */

        MPI_Send(u+((size-1)*sizeproc), sizeproc, MPI_DOUBLE, size-1,1, new_comm);
      }
    //*****************************************************************
    //* Distribution of r if domain is decomposed into second form 

    if(domain[4]==2 || domain[4]==3)
    {
      sizeproc_2= domain[3] * numgridpoints_x;
        
      for (int i = 0; i < size-1; ++i)
      {
        MPI_Send((r+ (i* sizeproc)),sizeproc,MPI_DOUBLE,i,0,new_comm);

      }
        MPI_Send(r +((size-1)* sizeproc),sizeproc_2,MPI_DOUBLE,size-1,0,new_comm);

        MPI_Send(u +((size-1)* sizeproc),sizeproc_2,MPI_DOUBLE,size-1,1,new_comm);
    }

    // //******************************************************************
    // // Distribution of r if domain is decomposed into third form 
    // if(domain[4]==3)
    // {
    //   sizeproc= domain[1] * numgridpoints_x;
    //   int sizeproc_2= domain[3] * numgridpoints_x;
    //   for (int i = 0; i < size-1; ++i)
    //   {
    //     MPI_Send((r+ (i* sizeproc)),domain[1]*numgridpoints_x,MPI_DOUBLE,i,0,new_comm);

    //   }
    //     MPI_Send(r +(size-1* sizeproc),sizeproc_2,MPI_DOUBLE,size-1,0,new_comm);
       
    // }

   }
  else
  {
    MPI_Status status1,status2;
    
    // Only the last process will receive u values as rest all are zero
    if(cart_rank == size-1) 
    { 
      if(domain[4]==1) 
      { 
      MPI_Recv(local_u,sizeproc,MPI_DOUBLE,0,1,new_comm,&status1);
      MPI_Recv(local_r,sizeproc,MPI_DOUBLE,0,0,new_comm,&status2);
      }
      else
      {
      MPI_Recv(local_u,sizeproc_2,MPI_DOUBLE,0,1,new_comm,&status1);
      MPI_Recv(local_r,sizeproc_2,MPI_DOUBLE,0,0,new_comm,&status2);  
      }
   }
   else  
   MPI_Recv(local_r,sizeproc,MPI_DOUBLE,0,0,new_comm,&status2); 

    // d is initially r
    for(int i=0; i<sizeproc; ++i)
    {
      local_d[i] = local_r[i];
    }
    
    std::cout<<"I am rank"<<cart_rank<<"and I have received my r"<<std::endl;
    for (int i = 0; i < sizeproc; ++i)
    {
      std::cout<<local_r[i]<<std::endl;
    }  
      
  }
    
}


//******************************
// Correct idex calculator Function. It calculates the inner points in any domain

void calInnerGridIndexes(const int &numGridX,const int &numGridY,std::vector<int>& ind) {
  
    for(int i=1; i<numGridY - 1; ++i) {
       for(int j=1 ;j<numGridX-1;++j) {
        ind.push_back(i*numGridX +j);
      }
    }
 }

//************************************






