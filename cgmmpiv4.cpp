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
inline void Boundary_Init(const int &numx,const int &numy,const double hx,double u[]);
inline void func_init(const int &numx,const int &numy,const double &hx,const double &hy,std::vector<double>& f);
inline double Compute_Matrix_Vec_Mult(const double &u_center,const double &u_left,const double &u_right,const double &u_up,const double &u_down,const double &constant,const double &hxsqinv,const double &hysqinv);
void Correct_Index_Cal(const int &numGridX,const int &numGridY,std::vector<int>& ind) ;
inline void write_sol(const std::vector<double>& u_temp,const double hx,const double hy,const int numx);
inline bool terminate(const double value);
inline double Scalar_Product(const int temp_r[],const int size);


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
  	int sizeproc=0;
  	int domain[5] ={0};
    double r[numgridpoints_x*numgridpoints_y]={0.0};
    double d[numgridpoints_x*numgridpoints_y]={0.0};
    double u[numgridpoints_x*numgridpoints_y]={0.0};
    
  	//***************************************
  	//** Domain Decompose *****************//
  	//***************************************
  	if (rank==0)
  	{
		  domaindecompose(numgridpoints_x,numgridpoints_y,size,domain);

  	}
    
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

    MPI_Cart_create(MPI_COMM_WORLD,ndims,dim,periods,reorder,&new_comm);
    MPI_Comm_rank(new_comm,&cart_rank);
    MPI_Cart_coords(new_comm,cart_rank,2,coords);
    MPI_Cart_shift(new_comm,1,1,&down,&up);
    //std::cout<<"this is my rank"<<cart_rank<<std::endl;
    //std::cout<<"This is my neighbours"<<up<<"  "<<down<<std::endl;

    //***********************************************
    //**** Calculation of Residual r by rank 0  ***//
    //***********************************************
    if(cart_rank==0)
    {
      //std::vector<double> u_temp(numgridpoints_x*numgridpoints_y,0.0);
      Boundary_Init(numgridpoints_x,numgridpoints_y,hx,u);
      /*
      for (int i = 0; i < numgridpoints_x*numgridpoints_y; ++i)
      {
        std::cout<<u[i]<<std::endl;
      }
      */

      //**********************************************************************
      //Forcing Conditins
      //**********************************************************************
        std::vector<double> f(numgridpoints_x*numgridpoints_y,0.0);
        func_init(numgridpoints_x, numgridpoints_y,hx,hy,f);
      //**********************************************************************

      //**********************************************************************
      //Residual array
      //**********************************************************************
        //double r[numgridpoints_x*numgridpoints_y]={0};
      //**********************************************************************

      //**********************************************************************
      //Correct Index Calculation
      //**********************************************************************
        std::vector<int> ind;
        Correct_Index_Cal(numgridpoints_x,numgridpoints_y,ind);
        for (size_t i = 0; i < ind.size(); i++) 
      //std::cout << "The corect indices are " <<ind[i]<< std::endl;
      
      // Computing the r by rank 0 ;

      double delta_r_1=0;
      for (size_t i = 0; i < ind.size(); i++) {
        double u_left= u[ind[i]-1];
        double u_right= u[ind[i]+1];
        double u_up= u[ind[i]+numgridpoints_x];
        double u_down = u[ind[i]-numgridpoints_x];
        double u_center= u[ind[i]];
        r[ind[i]]= f[ind[i]] - Compute_Matrix_Vec_Mult(u_center,u_left,u_right,u_up,u_down,constant,hxsqinv,hysqinv);
      }

      /*
      for (int i = 0; i < numgridpoints_x*numgridpoints_y; ++i)
      {
        std::cout<<r[i]<<std::endl;
      }
      */
      
    }
  	//********************************************
  	// Allocating memory in each processor *****//
  	//********************************************
  	// Memory in each processor allocated according to domain array 
    
    // Communicate 
    
    if(cart_rank == 0)
    {
      if (domain[2]==0)
      {
        sizeproc = domain[1] * numgridpoints_x;
        for (int i = 1; i < size; ++i)
        {
          MPI_Send(&sizeproc,1,MPI_INT,i,0,new_comm);
        }
      }
      else
      {
        sizeproc= domain[1] * numgridpoints_x;
        int sizeproc_2= domain[3] * numgridpoints_x;
        for (int i = 1; i < size-1; ++i)
        {
          MPI_Send(&sizeproc,1,MPI_INT,i,0,new_comm);
        }
          MPI_Send(&sizeproc_2,1,MPI_INT,size-1,0,new_comm);
      }
    }
	  else
	  {
  		MPI_Status status;
  		MPI_Recv(&sizeproc,1,MPI_INT,0,0,new_comm,&status);
  		//std::cout<<"I am Rank "<<rank<<" and I have received "<<sizeproc<<"memeory for computation"<<std::endl;
  	}

    // Memory allocation for each processor according to cartesian topology
    /*
    if(down<0 || up <0)
    {
      //std::cout<<"I am rank "<<cart_rank<<std::endl;
      //std::cout<<"I am allocating memeory    "<<sizeproc+numgridpoints_x<<" for computation"<<std::endl;
     
      //Allocating for r
      double r[sizeproc]={0};
      //Allocating for d
      double d[sizeproc+(1*numgridpoints_x)]={0};
    

    }
    else
    {
      //std::cout<<"I am rank "<<cart_rank<<std::endl;
      //std::cout<<"I am allocating memory    "<<sizeproc+(2*numgridpoints_x)<<"for computation"<<std::endl;
     
      //Allocating for r
      double r[sizeproc]={0};
      //Allocating for d
      double d[sizeproc+(2*numgridpoints_x)]={0};
    }
    */
    
    //******************************************************
    //*** Distribution of r into several processors ********
    //****************************************************** 

    //***************************************************************
    // Distribution of r if domain is decomposed in first form ******
  if(cart_rank==0)
  {
    if(domain[4]==1)
    {
      for (int i = 1; i < size; ++i)
      {
        /* Sending r  */
        MPI_Send((r+ (i* sizeproc)),domain[1]*numgridpoints_x,MPI_DOUBLE,i,0,new_comm);

      }
    }
    //*****************************************************************
    //* Distribution of r if domain is decomposed into second form 

    if(domain[4]==2)
    {
      sizeproc= domain[1] * numgridpoints_x;
      int sizeproc_2= domain[3] * numgridpoints_x;
        
      for (int i = 1; i < size-1; ++i)
      {
        MPI_Send((r+ (i* sizeproc)),domain[1]*numgridpoints_x,MPI_DOUBLE,i,0,new_comm);

      }
        MPI_Send((r +((size-1)* sizeproc)),sizeproc_2,MPI_DOUBLE,size-1,0,new_comm);
    }

    //******************************************************************
    // Distribution of r if domain is decomposed into third form 
    if(domain[4]==3)
    {
      /*
      for(int i =0 ; i<numgridpoints_x*numgridpoints_y;++i)
      {
        std::cout<<r[i]<<std::endl;
      }
      */
      sizeproc= domain[1] * numgridpoints_x;
      int sizeproc_2= domain[3] * numgridpoints_x;
      for (int i = 1; i < size-1; ++i)
      {
        MPI_Send((r+ (i* sizeproc)),domain[1]*numgridpoints_x,MPI_DOUBLE,i,0,new_comm);

      }
        //std::cout<<size-1<<std::endl;
        MPI_Send((r +((size-1)* sizeproc)),sizeproc_2,MPI_DOUBLE,size-1,0,new_comm);
       
    }

  }
  else
  {
    MPI_Status status;
    MPI_Recv(r,sizeproc,MPI_DOUBLE,0,0,new_comm,&status);
    
    //std::cout<<"I am rank"<<cart_rank<<"and I have received my r"<<std::endl;
    /*
    if(cart_rank==2)
    {
      for (int i = 0; i < sizeproc; ++i)
      {
        std::cout<<r[i]<<std::endl;
      }
    }
    */
    
    
  }  

    //******************************************************
    // Distribution of u into several processor ************
    //******************************************************
  
  if(cart_rank==0)
  {
     
    //****************************************************************
    // Distribution of u if domain is divided into first form 
    if(domain[4]==1)
    {
     
      for (int i = 1; i < size; ++i)
      {
        /* Sending r  */
        MPI_Send((u+ (i* sizeproc)),domain[1]*numgridpoints_x,MPI_DOUBLE,i,0,new_comm);

      }
    }

    //*****************************************************************
    //* Distribution of u if domain is decomposed into second form 

    if(domain[4]==2)
    {
      sizeproc= domain[1] * numgridpoints_x;
      int sizeproc_2= domain[3] * numgridpoints_x;
        
      for (int i = 1; i < size-1; ++i)
      {
        MPI_Send((u+ (i* sizeproc)),domain[1]*numgridpoints_x,MPI_DOUBLE,i,0,new_comm);

      }
        MPI_Send((u +((size-1)* sizeproc)),sizeproc_2,MPI_DOUBLE,size-1,0,new_comm);
    }

    //******************************************************************
    //* Distribution of u if domain is decomposed into third form 
    if(domain[4]==3)
    {
      sizeproc= domain[1] * numgridpoints_x;
      int sizeproc_2= domain[3] * numgridpoints_x;
      for (int i = 1; i < size-1; ++i)
      {
        MPI_Send((u+ (i* sizeproc)),domain[1]*numgridpoints_x,MPI_DOUBLE,i,0,new_comm);

      }
        MPI_Send((u +((size-1)* sizeproc)),sizeproc_2,MPI_DOUBLE,size-1,0,new_comm);
       
    }

  }
  else
  {
    MPI_Status status;
    MPI_Recv(u,sizeproc,MPI_DOUBLE,0,0,new_comm,&status);
    //std::cout<<" I have received size of "<<sizeproc<<std::endl;
    //std::cout<<"I am rank"<<cart_rank<<"and I have received my u"<<std::endl;
    /*
    if(cart_rank==2)
    {  
      for (int i = 0; i < sizeproc; ++i)
      {
        std::cout<<u[i]<<std::endl;
      }
    }
    */
    
  }

  //******************************************
  //****** Set all data arrays for rank 0 ****
  //******************************************
  // Rank 0 processor will be working on size_proc always ...whatever be the domain decomposition ....

  //************************************************************
  //**** CONJUCATE GRADIENT ALGORITHM **************************
  //************************************************************

    // communicate D 
    // we got z
    // we have to calcuate delta r 
    // 

  //************************************************
  //*** ASSIGNMENT OF D ****************************
  //************************************************
  // if I am 0th guy then 
  if(down< 0)
      {
        //std::cout<<"i am rank "<<cart_rank<<std::endl;
         for (int i = 0; i < sizeproc; ++i)
         {
           d[i]=r[i] ;// Assign r to d 
        //   std::cout<<d[i]<<std::endl;
         }
  
      }
  // This else covers 2 cases : interior processer and extreme up processor     
  else 
      {
        //std::cout<<"i am rank "<<cart_rank<<std::endl;
        for (int i = 0; i < sizeproc; ++i)
        {
          d[i+numgridpoints_x]= r[i];
          //std::cout<<d[i+numgridpoints_x]<<std::endl;
        }
      }
  
 //  if(down < 0)
 //  {
 //    for(int i=0;i<sizeproc; ++i)
 //    std::cout << d[i] << "  ";
 //    std::cout << "\n";
 // }

  // CG Loop 
  //for (int i = 0; i < num_iter; ++i)
  //{
    // Communicate D
    
   // MPI_Request request[6];
    MPI_Status status[3];
    // Send up  
    if(cart_rank == 0)
    {
      //std::cout<<" I am rank"<<cart_rank<<std::endl;
    //  std::cout << "My rank is : " <<  cart_rank << "and I am Sending to  " << up << std::endl;
      //std::cout<<std::endl;
      //MPI_Isend(up)
      // Which one to send 
      //MPI_Request request;
     //  std::cout <<" sizeproc is : " << sizeproc << 
     // "\n";
      MPI_Send(d + sizeproc-numgridpoints_x, numgridpoints_x, MPI_DOUBLE, up, 2,new_comm);
      //  std::cout << "d for rank 0 is \n";
      // for(int i=0;i<sizeproc+numgridpoints_x; ++i)
      // {
      //   std::cout << d[i] << std::endl;
      // }
     
    }
   // interrior point
    else if(up>0 && (cart_rank==1) )
    {
      // MPI_Request request;
       // It has one ghost cell so 
       MPI_Send(d + sizeproc, numgridpoints_x, MPI_DOUBLE, up, 2,new_comm);

    }
    
   // MPI_Barrier(new_comm);

    // Receive from down
    if(down>=0)
    {
     // std::cout<<" I am rank"<<cart_rank<<std::endl;
      //std::cout<<std::endl;
      //MPI_Request request;
      MPI_Recv(d,numgridpoints_x, MPI_DOUBLE, down,2,new_comm,&status[0]);
     // std::cout << "Source of 2 is " << status[0].MPI_SOURCE << "\n";
   // std::cout <<  " MPI_ERROR is: " << status[0].MPI_ERROR << "\n";
   
    }
    //MPI_Barrier(new_comm);
    
   // Each process will send the data downwards 
    // send down
   if(down>=0 && (cart_rank == 2) )
    {
      //std::cout<<" I am rank"<<cart_rank<<std::endl;
      std::cout<<std::endl;
      //MPI_Request request; 
      MPI_Send(d+numgridpoints_x,numgridpoints_x,MPI_DOUBLE,down,3,new_comm); 

      // std::cout << "I am rank " << cart_rank << " and I have sent the following data to "<< down << "\n";
      // for(int i=0; i<numgridpoints_x; i++)
      //   std::cout << d[numgridpoints_x+i] <<"\n";

    }
    
    //MPI_Barrier(new_comm);

    // Receive from up
    if(cart_rank ==0)
    {
      //MPI_Request request;
      MPI_Recv(d+sizeproc,numgridpoints_x, MPI_DOUBLE, up,3,new_comm,&status[1]);
      //std::cout <<  " MPI_ERROR is: " << status[0].MPI_ERROR << "\n";

      // std::cout << "I am rank " << cart_rank << " and I have received the following data from  "<< up << "\n";
      // for(int i=0; i<numgridpoints_x; i++)
      //   std::cout << d[sizeproc+i] <<"\n";
  
    }

   else if(up>0 && (cart_rank ==1) )
    {
      //MPI_Request request;
      MPI_Recv(d+sizeproc+numgridpoints_x,numgridpoints_x, MPI_DOUBLE, up,3,new_comm, &status[2]);
      //std::cout <<  " MPI_ERROR is: " << status[0].MPI_ERROR << "\n";
   
    }

   // MPI_Barrier(new_comm);

   // //  if(cart_rank ==1)
   // //  {
   // //    std::cout << " printing d for processor 1\n"; 
   // //    for(int i=0; i<sizeproc+2*numgridpoints_x; ++i) 
   // //    std::cout << d[i] << "\n";

   // }
  
  //MPI_Waitall(6,request,status);
    
  //}
    
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
    if(numy%(size-1)!=0 && (!((numy-((numy/(size-1))*(size-1))) == 1 )))
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

inline void Boundary_Init(const int &numx,const int &numy,const double hx,double u[]){
  // Implementig the boundary value
  for (int i = (numy-1) * numx; i < (numx * numy) ; ++i ) {
      u[i] = (con_sinh* sin(2 * PI * (hx *(i - ((numy - 1) * numx)))));
      //std::cout<<"The value of u is"<<u[i]<<std::endl;
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
inline double Scalar_Product(const int temp_r[],const int size){
  // Calculating the norm of the vector
  double sum(0);
  for (size_t i = 0; i < size; i++) {
    // Computing the Norm
    sum+= temp_r[i] * temp_r[i];

  }
  return (sum);
}


//******************************
// Correct idex calculator Function 

void Correct_Index_Cal(const int &numGridX,const int &numGridY,std::vector<int>& ind) {
  // Calculating the inner points in any domain
  bool toggle = false;
    for(int i=1; i<numGridY - 1; ++i)
    {
       toggle = !toggle;
       for(int j=1 ;j<numGridX-1;j+=2)
       {
         if(toggle)
            {
              ind.push_back(i*numGridX +j);
              if((numGridX % 2) == 1)
               {
                 if (!((j+2) % numGridX == 0)) ind.push_back(1+i*numGridX +j);
               }
               else  ind.push_back(1+i*numGridX +j);
            }
          else
            {
              ind.push_back(i*numGridX +j);
              if((numGridX % 2) == 1)
               {
               if (!((j+2) % numGridX == 0)) ind.push_back(1+ i*numGridX +j);
               }
               else ind.push_back(1+ i*numGridX +j);
            }

       }
    }

}
//************************************






