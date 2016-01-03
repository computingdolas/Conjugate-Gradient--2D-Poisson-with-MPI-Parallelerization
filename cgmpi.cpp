//*************************************************************************//
//This Program Solves 2-D Poisson's Equation with Conjucate Gradient Algorithm and MPI Parallelisization on predefined domain
//*************************************************************************//
// 
//*********************************************//
//Includes
//*********************************************//
#include <iostream>
#include <stdlib.h>
#include <string>
//#include <array>
#include <vector>
#include <cmath>
#include <fstream>
//#include <mpi.h>

//********************************************//
// Global Variables
const double PI = 4.0 * atan(1.0);
const double con_sin = sinh(2*PI);
double tol = 0.000001;     
//********************************************//

//***********************************************
//Function Declaration
//***********************************************
void Boundary_Init(const int numx, const int numy,const double hx,std::vector<double>& u);
void funcInit(const int numx, const int numy,const double hx,const double hy,std::vector<double>& f);
double Scalar_Product(std::vector<double>& temp_r);
void CG_Paralle_Solver(int numgridpoints_x, int numgridpoints_y , std::vector<double>& u,const std::vector<double>& f,std::vector<double>& r,const std::vector<int>& ind,const double hxsqinv,const double hysqinv,const double constant, int num_iter,const int numx);
void Correct_Index_Cal(const int numGridX,const int numGridY,std::vector<int>& ind);
double Compute_Matrix_Vec_Mult(const double u_center,const double u_left,const double u_right,const double u_up,const double u_down,const double constant,const double hxsqinv,const double hysqinv);
inline bool terminate(const double value);
inline void solution(const std::vector<double>& u, const int numgridpoints_x);
//***********************************************




int main(int argc, char const *argv[]) {

  if (argc!=4) {
    /* Checking the Number of arguments are per the requirement*/
    std::cout << "Aborting the Program , Please check the number of the input arguments" << std::endl;
    exit(1);
  }
  else
    std::cout << "OK .....Parameters Taken" << std::endl;


//**********************************************************************
//Variable Declaration
//**********************************************************************
  std::string s1 = argv[1],s2= argv[2],s3 = argv[3];
  int numx_interval = std::stoi(s1) ;
  int numy_interval = std::stoi(s2);
  int num_iter = std::stoi(s3);
  int numgridpoints_x = numx_interval+1;
  int numgridpoints_y = numy_interval+1;
  const double hx  = 2.0 / numx_interval;
  const double hy = 1.0 / numy_interval;
  const double hxsq = hx * hx ;
  const double hysq = hy * hy ;
  const double hxsqinv = 1.0/ hxsq;
  const double hysqinv = 1.0/ hysq;
  const double K = 2 * PI;
  const double constant = (2*hxsqinv) + (2*hysqinv) + (K*K);
 // const double constinv = 1.0/ constant;
//**********************************************************************

//**********************************************************************
//Grid Generation and Boundary value initilization
//**********************************************************************
  std::vector<double> u(numgridpoints_x*numgridpoints_y,0.0);
  Boundary_Init(numgridpoints_x,numgridpoints_y,hx,u);
//**********************************************************************

//**********************************************************************
//Forcing Conditins
//**********************************************************************
  std::vector<double> f(numgridpoints_x*numgridpoints_y,0.0);
  funcInit(numgridpoints_x, numgridpoints_y,hx,hy,f);
//**********************************************************************

//**********************************************************************
//Residual array
//**********************************************************************
  std::vector<double> r(numgridpoints_x*numgridpoints_y,0.0);
//**********************************************************************

//**********************************************************************
//Correct Index Calculation
//**********************************************************************
  std::vector<int> ind;
  Correct_Index_Cal(numgridpoints_x,numgridpoints_y,ind);
  for (size_t i = 0; i < ind.size(); i++) {
    //std::cout << "The corect indices are " <<ind[i]<< std::endl;
  }

//**********************************************************************
//Conjucate Gradient Parallel Solver
//**********************************************************************
  CG_Paralle_Solver(numgridpoints_x, numgridpoints_y, u,f,r,ind,hxsqinv,hysqinv,constant,num_iter,numgridpoints_x);
//**********************************************************************

//**********************************************************************
//Saving Solution obtained
//**********************************************************************
  solution(u,numgridpoints_x);
  std::cout << "The Solution has been written in solution.txt" << std::endl;
//**********************************************************************


  return 0;
}
//*****************************************************
void Boundary_Init(int numx,int numy,const double hx, std::vector<double>& u){
  // Implementig the boundary value
  for (int i = (numy-1) * numx; i < (numx * numy) ; ++i ) {
      u[i] = (con_sin * sin(2 * PI * (hx *(i - ((numy - 1) * numx)))));
  }
  std::cout << "Boundary values initialized and cos_sin is  " << con_sin << std::endl;

//*****************************************************
}
void funcInit(int numx,int numy,const double hx,const double hy,std::vector<double>& f){
  // Updating the function value on any domain
  double x,y;
  for (int  i = 0; i < numx * numy; ++i) {
    x = hx * (i % numx);
    y = hy * (i / numx); 
    f[i] = 4 * PI *PI * sin(2 * PI * x) * sinh(2* PI * y );
    //std::cout << "The value of f is"<<f[i]<< std::endl;

  }
}
//*****************************************************
double Scalar_Product(std::vector<double>& temp_r){
  // Calculating the norm of the vector
  size_t size= temp_r.size();
  double sum(0.0);
  for (size_t i = 0; i < size; i++) {
    // Computing the Norm
    sum+= temp_r[i] * temp_r[i];

  }
  return (sum);
}
//*****************************************************
void CG_Paralle_Solver(int numgridpoints_x, int numgridpoints_y, std::vector<double>& u, const std::vector<double>& f, std::vector<double>& r,const std::vector<int>& ind,const double hxsqinv,const double hysqinv,const double constant,const int num_iter,const int numx) {

  // Computing the Residual
  for (size_t i = 0; i < ind.size(); ++i) {
    double u_left= u[ind[i]-1];
    double u_right= u[ind[i]+1];
    double u_up= u[ind[i]+numx];
    double u_down = u[ind[i]-numx];
    double u_center= u[ind[i]];
    r[ind[i]]= f[ind[i]] - Compute_Matrix_Vec_Mult(u_center,u_left,u_right,u_up,u_down,constant,hxsqinv,hysqinv);
  }
 // int numGridPoints = numgridpoints_x*numgridpoints_y;
  

  // Computing the norm
  double delta_r=Scalar_Product(r);  // include sqrt in the Scalar_Product and change the name of this function
  //std::cout << "The norm of the function is" <<delta_r<< std::endl;
  std::cout << "Scalar Product for, nx= " << numgridpoints_x << " and ny= " << numgridpoints_y << " is: "  << delta_r;


  // Check for terminating conditions
  if(terminate(sqrt(delta_r))) return;

  //allocate d=r
  std::vector<double> d=r;

  //create d
  std::vector<double> z(d.size(),0);

  static int count=0;
  for (int j = 0; j < num_iter; j++) {   // replace with while loop
    //count++;
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

  // std::cout << "\nglobal_scalar_d_z for iter: " << count+1 << "is  " << sum << std::endl;

    // Computing Alpha
    double suminv= 1.0/sum;      // no need to define suminv
    double alpha= suminv* delta_r;

   // std::cout << "\nalpha for iter " << count + 1 <<" is: " << alpha << std::endl;

    // u+ alpha*d

    for (size_t i = 0; i < u.size(); i++) {
      u[i]+= alpha*d[i];
    }

    // r= r- alpha*z

    for (size_t i = 0; i < r.size(); i++) {
      r[i]-= alpha*z[i];
    }

    double delta_r_1= Scalar_Product(r);
    std::cout << "delta_1 for iter : " << count+1 << " is : " << delta_r_1 << "\n";
    if(terminate(sqrt(delta_r_1)))
      break;

    // beta = delta_r_1/delta_r;
    double delta_r_inv = 1/delta_r;    // no need to define delta_r_inv and all the variables to be defined outside for loop
    double beta= delta_r_1 * delta_r_inv;
    //std::cout << "\nbeta for iter " << count + 1 <<" is: " << beta << std::endl;


    for (size_t i = 0; i < d.size(); i++) {
      d[i]= r[i] + beta*d[i];
    }

    delta_r=delta_r_1;
    count++;

  }

}
double Compute_Matrix_Vec_Mult(const double u_center,const double u_left,const double u_right,const double u_up,const double u_down,const double constant,const double hxsqinv,const double hysqinv){

  return (  (u_center * (constant) - (  ((hxsqinv)*(u_left + u_right))   + (hysqinv)*( u_up + u_down)  ) )   );
}

inline bool terminate(const double value){
  if(abs(value) <tol){
    std::cout << "The solution is converged...exiting" << std::endl;
    std::cout << "The residual is " <<value<< std::endl;
    //exit(0);
    return true;
  }
  else
    return false;
}
//*****************************************************

inline void solution(const std::vector<double>& u, const int numgridpoints_x){
  std::ofstream file_;
  file_.open("solution.txt");

  int x,y;

  for (size_t i = 0; i < u.size(); i++) {

    x=i%numgridpoints_x;
    y= i/numgridpoints_x;

    file_<< x<< "\t" << y << "\t" << u[i] << "\n";
  }

  file_.close();

}

//*****************************************************
void Correct_Index_Cal(const int numGridX,const int numGridY,std::vector<int>& ind) {
  // Calculating the inner points in any domain
  
    for(int i=1; i<numGridY - 1; ++i) {

       for(int j=1 ;j<numGridX-1; ++j) {
         ind.push_back(i*numGridX +j);
        }    

    }
 }   
//**************************************************************
