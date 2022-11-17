
//////////////////////////////////////////////////////////////////////////////////////////
// miniCFD
// Author: Omitted
//////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <ctime>
#include <iostream>

const double pi        = 3.14159265358979323846264338327;   //Pi
const double grav      = 9.8;                               //Gravitational acceleration (m / s^2)
const double cp        = 1004.;                             //Specific heat of dry air at constant pressure
const double cv        = 717.;                              //Specific heat of dry air at constant volume
const double rd        = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
const double p0        = 1.e5;                              //Standard pressure at the surface in Pascals
const double C0        = 27.5629410929725927310572984382;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
const double gamm      = 1.40027894002789401278940017893;   //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
const double xlen      = 2.e4;    //Length of the domain in the x-direction (meters)
const double zlen      = 1.e4;    //Length of the domain in the z-direction (meters)
const double hv   = 0.25;     //How strong to diffuse the solution: hv \in [0:1]
const double cfl       = 1.50;    //"Courant, Friedrichs, Lewy" number (for numerical stability)
const double max_speed = 450;        //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
const int hs        = 2;          //"Halo" size: number of cells beyond the MPI tasks's domain needed for a full "stencil" of information for reconstruction
const int cfd_size = 4;          //Size of the stencil used for interpolation

//Parameters for indexing and flags
const int NUM_VARS = 4;           //Number of fluid state variables
const int POS_DENS  = 0;           //index for density ("rho")
const int POS_UMOM  = 1;           //index for momentum in the x-direction ("rho * u")
const int POS_WMOM  = 2;           //index for momentum in the z-direction ("rho * w")
const int POS_RHOT  = 3;           //index for density * potential temperature ("rho * theta")
const int DIR_X = 1;              //Integer constant to express that this operation is in the x-direction
const int DIR_Z = 2;              //Integer constant to express that this operation is in the z-direction
enum test_cases {CONFIG_IN_TEST1, CONFIG_IN_TEST2, CONFIG_IN_TEST3,
    CONFIG_IN_TEST4, CONFIG_IN_TEST5, CONFIG_IN_TEST6 };

const int nqpoints = 3;
double qpoints [] = { 0.112701665379258311482074460012E0 , 0.510000000000000000000000000000E0 , 0.887298334621741688517926529880E0 };
double qweights[] = { 0.277777777777777777777777777778E0 , 0.444444444444444444444444444445E0 , 0.277777777777777777777777777786E0 };

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the coure of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double sim_time;              //total simulation time in seconds
double output_freq;           //frequency to perform output in seconds
double dt;                    //Model time step (seconds)
int    nnx, nnz;                //Number of local grid cells in the x- and z- dimensions for this MPI task
double dx, dz;                //Grid space length in x- and z-dimension (meters)
int    nx_cfd, nz_cfd;      //Number of total grid cells in the x- and z- dimensions
int    i_beg, k_beg;          //beginning index in the x- and z-directions for this MPI task
int    nranks, myrank;        //Number of MPI ranks and my rank id
int    masterproc;            //Am I the master process (rank == 0)?
int    config_spec;         //Which data initialization to use
double *cfd_dens_cell;         //density (vert cell avgs).   Dimensions: (1-hs:nnz+hs)
double *cfd_dens_theta_cell;   //rho*t (vert cell avgs).     Dimensions: (1-hs:nnz+hs)
double *cfd_dens_int;          //density (vert cell interf). Dimensions: (1:nnz+1)
double *cfd_dens_theta_int;    //rho*t (vert cell interf).   Dimensions: (1:nnz+1)
double *cfd_pressure_int;      //press (vert cell interf).   Dimensions: (1:nnz+1)

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double etime;                 //Elapsed model time
double output_counter;        //Helps determine when it's time to do output
//Runtime variable arrays
double *state;                //Fluid state.             Dimensions: (1-hs:nnx+hs,1-hs:nnz+hs,NUM_VARS)
double *state_tmp;            //Fluid state.             Dimensions: (1-hs:nnx+hs,1-hs:nnz+hs,NUM_VARS)
double *flux;                 //Cell interface fluxes.   Dimensions: (nnx+1,nnz+1,NUM_VARS)
double *tend;                 //Fluid state tendencies.  Dimensions: (nnx,nnz,NUM_VARS)
int    num_out = 0;           //The number of outputs performed so far
int    direction_switch = 1;
double mass0, te0;            //Initial domain totals for mass and total energy  
double mass , te ;            //Domain totals for mass and total energy  

//How is this not in the standard?!
double dmin( double a , double b ) { if (a<b) {return a;} else {return b;} };


//Declaring the functions defined after "main"
void   initialize                 ( int *argc , char ***argv );
void   finalize             ( );
void   testcase6            ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   testcase5      ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   testcase4           ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   testcase3       ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   testcase2              ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   testcase1            ( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht );
void   const_theta    ( double z                   , double &r , double &t );
void   const_bvfreq   ( double z , double bv_freq0 , double &r , double &t );

double sample_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad );
void   output               ( double *state , double etime );
void   do_timestep     ( double *state , double *state_tmp , double *flux , double *tend , double dt );
void   do_semi_step   ( double *state_init , double *state_forcing , double *state_out , double dt , int dir , double *flux , double *tend );
void   do_dir_x ( double *state , double *flux , double *tend );
void   do_dir_z ( double *state , double *flux , double *tend );
void   exchange_border_x    ( double *state );
void   exchange_border_z    ( double *state );
void   do_results           ( double &mass , double &te );

//Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
void do_timestep( double *state , double *state_tmp , double *flux , double *tend , double dt ) {
  if (direction_switch) {
    //x-direction first
    do_semi_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend );
    do_semi_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend );
    do_semi_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend );
    //z-direction second
    do_semi_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend );
    do_semi_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend );
    do_semi_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend );
  } else {
    //z-direction second
    do_semi_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend );
    do_semi_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend );
    do_semi_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend );
    //x-direction first
    do_semi_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend );
    do_semi_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend );
    do_semi_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend );
  }
  if (direction_switch) { direction_switch = 0; } else { direction_switch = 1; }
}


//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void do_semi_step( double *state_init , double *state_forcing , double *state_out , double dt , int dir , double *flux , double *tend ) {
  int i, k, ll, inds, indt;
  if        (dir == DIR_X) {
    //Set the halo values for this MPI task's fluid state in the x-direction
    exchange_border_x(state_forcing);
    //Compute the time tendencies for the fluid state in the x-direction
    do_dir_x(state_forcing,flux,tend);
  } else if (dir == DIR_Z) {
    //Set the halo values for this MPI task's fluid state in the z-direction
    exchange_border_z(state_forcing);
    //Compute the time tendencies for the fluid state in the z-direction
    do_dir_z(state_forcing,flux,tend);
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Apply the tendencies to the fluid state
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nnz; k++) {
      for (i=0; i<nnx; i++) {
        inds = (k+hs)*(nnx+2*hs) + ll*(nnz+2*hs)*(nnx+2*hs) + i+hs;
        indt = ll*nnz*nnx + k*nnx + i;
        state_out[inds] = state_init[inds] + dt * tend[indt];
      }
    }
  }
}


//Compute the time tendencies of the fluid state using forcing in the x-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the x-direction (including viscosity)
//Then, compute the tendencies using those fluxes
void do_dir_x( double *state , double *flux , double *tend ) {
  int    i,k,ll,s,inds,indf1,indf2,indt;
  double r,u,w,t,p, stencil[4], d_vals[NUM_VARS], vals[NUM_VARS], v_coef;
  //Compute the hyperviscosity coeficient
  v_coef = -hv * dx / (16*dt);
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (k=0; k<nnz; k++) {
    for (i=0; i<nnx+1; i++) {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s < cfd_size; s++) {
          inds = ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+s;
          stencil[s] = state[inds];
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[POS_DENS] + cfd_dens_cell[k+hs];
      u = vals[POS_UMOM] / r;
      w = vals[POS_WMOM] / r;
      t = ( cfd_dens_theta_cell[k+hs] + vals[POS_RHOT] ) / r;
      p = pow((r*t),gamm)*C0;

      //Compute the flux vector
      flux[POS_DENS*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u     - v_coef*d_vals[POS_DENS];
      flux[POS_UMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u*u+p - v_coef*d_vals[POS_UMOM];
      flux[POS_WMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u*w   - v_coef*d_vals[POS_WMOM];
      flux[POS_RHOT*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*u*t   - v_coef*d_vals[POS_RHOT];
    }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nnz; k++) {
      for (i=0; i<nnx; i++) {
        indt  = ll* nnz   * nnx    + k* nnx    + i  ;
        indf1 = ll*(nnz+1)*(nnx+1) + k*(nnx+1) + i  ;
        indf2 = ll*(nnz+1)*(nnx+1) + k*(nnx+1) + i+1;
        tend[indt] = -( flux[indf2] - flux[indf1] ) / dx;
      }
    }
  }
}


//Compute the time tendencies of the fluid state using forcing in the z-direction
//Since the halos are set in a separate routine, this will not require MPI
//First, compute the flux vector at each cell interface in the z-direction (including viscosity)
//Then, compute the tendencies using those fluxes
void do_dir_z( double *state , double *flux , double *tend ) {
  int    i,k,ll,s, inds, indf1, indf2, indt;
  double r,u,w,t,p, stencil[4], d_vals[NUM_VARS], vals[NUM_VARS], v_coef;
  //Compute the viscosity coeficient
  v_coef = -hv * dz / (16*dt);
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
  for (k=0; k<nnz+1; k++) {
    for (i=0; i<nnx; i++) {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll=0; ll<NUM_VARS; ll++) {
        for (s=0; s<cfd_size; s++) {
          inds = ll*(nnz+2*hs)*(nnx+2*hs) + (k+s)*(nnx+2*hs) + i+hs;
          stencil[s] = state[inds];
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
        //First-order-accurate interpolation of the third spatial derivative of the state
        d_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[POS_DENS] + cfd_dens_int[k];
      u = vals[POS_UMOM] / r;
      w = vals[POS_WMOM] / r;
      t = ( vals[POS_RHOT] + cfd_dens_theta_int[k] ) / r;
      p = C0*pow((r*t),gamm) - cfd_pressure_int[k];
      //Enforce vertical boundary condition and exact mass conservation
      if (k == 0 || k == nnz) {
        w                = 0;
        d_vals[POS_DENS] = 0;
      }

      //Compute the flux vector with viscosity
      flux[POS_DENS*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w     - v_coef*d_vals[POS_DENS];
      flux[POS_UMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w*u   - v_coef*d_vals[POS_UMOM];
      flux[POS_WMOM*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w*w+p - v_coef*d_vals[POS_WMOM];
      flux[POS_RHOT*(nnz+1)*(nnx+1) + k*(nnx+1) + i] = r*w*t   - v_coef*d_vals[POS_RHOT];
    }
  }

  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Use the fluxes to compute tendencies for each cell
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nnz; k++) {
      for (i=0; i<nnx; i++) {
        indt  = ll* nnz   * nnx    + k* nnx    + i  ;
        indf1 = ll*(nnz+1)*(nnx+1) + (k  )*(nnx+1) + i;
        indf2 = ll*(nnz+1)*(nnx+1) + (k+1)*(nnx+1) + i;
        tend[indt] = -( flux[indf2] - flux[indf1] ) / dz;
        if (ll == POS_WMOM) {
          inds = POS_DENS*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
          tend[indt] = tend[indt] - state[inds]*grav;
        }
      }
    }
  }
}



//Set this MPI task's halo values in the x-direction. This routine will require MPI
void exchange_border_x( double *state ) {
  int k, ll, ind_r, ind_u, ind_t, i;
  double z;
  ////////////////////////////////////////////////////////////////////////
  // TODO: EXCHANGE HALO VALUES WITH NEIGHBORING MPI TASKS
  // (1) give    state(1:hs,1:nnz,1:NUM_VARS)       to   my left  neighbor
  // (2) receive state(1-hs:0,1:nnz,1:NUM_VARS)     from my left  neighbor
  // (3) give    state(nnx-hs+1:nnx,1:nnz,1:NUM_VARS) to   my right neighbor
  // (4) receive state(nnx+1:nnx+hs,1:nnz,1:NUM_VARS) from my right neighbor
  ////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////
  // DELETE THE SERIAL CODE BELOW AND REPLACE WITH MPI
  //////////////////////////////////////////////////////
  for (ll=0; ll<NUM_VARS; ll++) {
    for (k=0; k<nnz; k++) {
      state[ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + 0      ] = state[ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + nnx+hs-2];
      state[ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + 1      ] = state[ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + nnx+hs-1];
      state[ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + nnx+hs  ] = state[ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + hs     ];
      state[ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + nnx+hs+1] = state[ll*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + hs+1   ];
    }
  }
  ////////////////////////////////////////////////////

  if (config_spec == CONFIG_IN_TEST6) {
    if (myrank == 0) {
      for (k=0; k<nnz; k++) {
        for (i=0; i<hs; i++) {
          z = (k_beg + k+0.5)*dz;
          if (fabs(z-3*zlen/4) <= zlen/16) {
            ind_r = POS_DENS*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i;
            ind_u = POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i;
            ind_t = POS_RHOT*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i;
            state[ind_u] = (state[ind_r]+cfd_dens_cell[k+hs]) * 50.;
            state[ind_t] = (state[ind_r]+cfd_dens_cell[k+hs]) * 298. - cfd_dens_theta_cell[k+hs];
          }
        }
      }
    }
  }
}


//Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
//decomposition in the vertical direction
void exchange_border_z( double *state ) {
  int          i, ll;
  const double mnt_width = xlen/8;
  double       x, xloc, mnt_deriv;
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  for (ll=0; ll<NUM_VARS; ll++) {
    for (i=0; i<nnx+2*hs; i++) {
      if (ll == POS_WMOM) {
        state[ll*(nnz+2*hs)*(nnx+2*hs) + (0      )*(nnx+2*hs) + i] = 0.;
        state[ll*(nnz+2*hs)*(nnx+2*hs) + (1      )*(nnx+2*hs) + i] = 0.;
        state[ll*(nnz+2*hs)*(nnx+2*hs) + (nnz+hs  )*(nnx+2*hs) + i] = 0.;
        state[ll*(nnz+2*hs)*(nnx+2*hs) + (nnz+hs+1)*(nnx+2*hs) + i] = 0.;
        //Impose the vertical momentum effects of an artificial cos^2 mountain at the lower boundary
        if (config_spec == CONFIG_IN_TEST3) {
          x = (i_beg+i-hs+0.5)*dx;
          if ( fabs(x-xlen/4) < mnt_width ) {
            xloc = (x-(xlen/4)) / mnt_width;
            //Compute the derivative of the fake mountain
            mnt_deriv = -pi*cos(pi*xloc/2)*sin(pi*xloc/2)*10/dx;
            //w = (dz/dx)*u
            state[POS_WMOM*(nnz+2*hs)*(nnx+2*hs) + (0)*(nnx+2*hs) + i] = mnt_deriv*state[POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + hs*(nnx+2*hs) + i];
            state[POS_WMOM*(nnz+2*hs)*(nnx+2*hs) + (1)*(nnx+2*hs) + i] = mnt_deriv*state[POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + hs*(nnx+2*hs) + i];
          }
        }
      } else {
        state[ll*(nnz+2*hs)*(nnx+2*hs) + (0      )*(nnx+2*hs) + i] = state[ll*(nnz+2*hs)*(nnx+2*hs) + (hs     )*(nnx+2*hs) + i];
        state[ll*(nnz+2*hs)*(nnx+2*hs) + (1      )*(nnx+2*hs) + i] = state[ll*(nnz+2*hs)*(nnx+2*hs) + (hs     )*(nnx+2*hs) + i];
        state[ll*(nnz+2*hs)*(nnx+2*hs) + (nnz+hs  )*(nnx+2*hs) + i] = state[ll*(nnz+2*hs)*(nnx+2*hs) + (nnz+hs-1)*(nnx+2*hs) + i];
        state[ll*(nnz+2*hs)*(nnx+2*hs) + (nnz+hs+1)*(nnx+2*hs) + i] = state[ll*(nnz+2*hs)*(nnx+2*hs) + (nnz+hs-1)*(nnx+2*hs) + i];
      }
    }
  }
}


void initialize( int *argc , char ***argv ) {
  int    i, k, ii, kk, ll, inds;
  double x, z, r, u, w, t, hr, ht;

  //Set the cell grid size
  dx = xlen / nx_cfd;
  dz = zlen / nz_cfd;

  /////////////////////////////////////////////////////////////
  // BEGIN MPI DUMMY SECTION
  // TODO: (1) GET NUMBER OF MPI RANKS
  //       (2) GET MY MPI RANK ID (RANKS ARE ZERO-BASED INDEX)
  //       (3) COMPUTE MY BEGINNING "I" INDEX (1-based index)
  //       (4) COMPUTE HOW MANY X-DIRECTION CELLS MY RANK HAS
  //       (5) FIND MY LEFT AND RIGHT NEIGHBORING RANK IDs
  /////////////////////////////////////////////////////////////
  i_beg = 0;
  nnx = nx_cfd;
  nranks = 1;
  myrank = 0;
  //////////////////////////////////////////////
  // END MPI DUMMY SECTION
  //////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  //Vertical direction isn't MPI-ized, so the rank's local values = the global values
  k_beg = 0;
  nnz = nz_cfd;
  masterproc = (myrank == 0);

  //Allocate the model data
  state              = (double *) malloc( (nnx+2*hs)*(nnz+2*hs)*NUM_VARS*sizeof(double) );
  state_tmp          = (double *) malloc( (nnx+2*hs)*(nnz+2*hs)*NUM_VARS*sizeof(double) );
  flux               = (double *) malloc( (nnx+1)*(nnz+1)*NUM_VARS*sizeof(double) );
  tend               = (double *) malloc( nnx*nnz*NUM_VARS*sizeof(double) );
  cfd_dens_cell       = (double *) malloc( (nnz+2*hs)*sizeof(double) );
  cfd_dens_theta_cell = (double *) malloc( (nnz+2*hs)*sizeof(double) );
  cfd_dens_int        = (double *) malloc( (nnz+1)*sizeof(double) );
  cfd_dens_theta_int  = (double *) malloc( (nnz+1)*sizeof(double) );
  cfd_pressure_int    = (double *) malloc( (nnz+1)*sizeof(double) );

  //Define the maximum stable time step based on an assumed maximum wind speed
  dt = dmin(dx,dz) / max_speed * cfl;
  //Set initial elapsed model time 
  etime = 0.;

  //If I'm the master process in MPI, display some grid information
  if (masterproc) {
    printf( "nx_cfd, nz_cfd: %d %d\n", nx_cfd, nz_cfd);
    printf( "dx,dz: %lf %lf\n",dx,dz);
    printf( "dt: %lf\n",dt);
  }

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  for (k=0; k<nnz+2*hs; k++) {
    for (i=0; i<nnx+2*hs; i++) {
      //Initialize the state to zero
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
        state[inds] = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a balance + temperature perturbation
      for (kk=0; kk<nqpoints; kk++) {
        for (ii=0; ii<nqpoints; ii++) {
          //Compute the x,z location within the global domain based on cell and quadrature index
          x = (i_beg + i-hs+0.5)*dx + (qpoints[ii]-0.5)*dx;
          z = (k_beg + k-hs+0.5)*dz + (qpoints[kk]-0.5)*dz;

          //Set the fluid state based on the user's specification
          switch(config_spec){
            case CONFIG_IN_TEST1: 
              testcase1(x,z,r,u,w,t,hr,ht); 
              break;
            case CONFIG_IN_TEST2: 
              testcase2(x,z,r,u,w,t,hr,ht); 
              break;
            case CONFIG_IN_TEST3: 
              testcase3(x,z,r,u,w,t,hr,ht); 
              break;
            case CONFIG_IN_TEST4: 
              testcase4(x,z,r,u,w,t,hr,ht); 
              break;
            case CONFIG_IN_TEST5: 
              testcase5(x,z,r,u,w,t,hr,ht); 
              break;
            case CONFIG_IN_TEST6: 
              testcase6(x,z,r,u,w,t,hr,ht); 
              break;
          }

          //Store into the fluid state array
          inds = POS_DENS*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
          state[inds] = state[inds] + r                         * qweights[ii]*qweights[kk];
          inds = POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
          state[inds] = state[inds] + (r+hr)*u                  * qweights[ii]*qweights[kk];
          inds = POS_WMOM*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
          state[inds] = state[inds] + (r+hr)*w                  * qweights[ii]*qweights[kk];
          inds = POS_RHOT*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
          state[inds] = state[inds] + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk];
        }
      }
      for (ll=0; ll<NUM_VARS; ll++) {
        inds = ll*(nnz+2*hs)*(nnx+2*hs) + k*(nnx+2*hs) + i;
        state_tmp[inds] = state[inds];
      }
    }
  }
  //Compute the background state over vertical cell averages
  for (k=0; k<nnz+2*hs; k++) {
    cfd_dens_cell      [k] = 0.;
    cfd_dens_theta_cell[k] = 0.;
    for (kk=0; kk<nqpoints; kk++) {
      z = (k_beg + k-hs+0.5)*dz;
      //Set the fluid state based on the user's specification
      if (config_spec == CONFIG_IN_TEST1      ) { testcase1      (0.,z,r,u,w,t,hr,ht); }
      if (config_spec == CONFIG_IN_TEST2        ) { testcase2        (0.,z,r,u,w,t,hr,ht); }
      if (config_spec == CONFIG_IN_TEST3       ) { testcase3 (0.,z,r,u,w,t,hr,ht); }
      if (config_spec == CONFIG_IN_TEST4     ) { testcase4     (0.,z,r,u,w,t,hr,ht); }
      if (config_spec == CONFIG_IN_TEST5) { testcase5(0.,z,r,u,w,t,hr,ht); }
      if (config_spec == CONFIG_IN_TEST6      ) { testcase6      (0.,z,r,u,w,t,hr,ht); }
      cfd_dens_cell      [k] = cfd_dens_cell      [k] + hr    * qweights[kk];
      cfd_dens_theta_cell[k] = cfd_dens_theta_cell[k] + hr*ht * qweights[kk];
    }
  }
  //Compute the background state at vertical cell interfaces
  for (k=0; k<nnz+1; k++) {
    z = (k_beg + k)*dz;
    if (config_spec == CONFIG_IN_TEST1      ) { testcase1      (0.,z,r,u,w,t,hr,ht); }
    if (config_spec == CONFIG_IN_TEST2        ) { testcase2        (0.,z,r,u,w,t,hr,ht); }
    if (config_spec == CONFIG_IN_TEST3       ) { testcase3 (0.,z,r,u,w,t,hr,ht); }
    if (config_spec == CONFIG_IN_TEST4     ) { testcase4     (0.,z,r,u,w,t,hr,ht); }
    if (config_spec == CONFIG_IN_TEST5) { testcase5(0.,z,r,u,w,t,hr,ht); }
    if (config_spec == CONFIG_IN_TEST6      ) { testcase6      (0.,z,r,u,w,t,hr,ht); }
    cfd_dens_int      [k] = hr;
    cfd_dens_theta_int[k] = hr*ht;
    cfd_pressure_int  [k] = C0*pow((hr*ht),gamm);
  }
}


//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase6( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
}


//Initialize a density current (falling cold thermal that propagates along the model bottom)
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase5( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_cosine(x,z,-20. ,xlen/2,5000.,4000.,2000.);
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase4( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  // call random_number(u);
  // call random_number(w);
  // u = (u-0.5)*20;
  // w = (w-0.5)*20;
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase3( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_bvfreq(z,0.02,hr,ht);
  r = 0.;
  t = 0.;
  u = 15.;
  w = 0.;
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase2( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_cosine(x,z, 3. ,xlen/2,2000.,2000.,2000.);
}


//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background density and potential temperature at that location
void testcase1( double x , double z , double &r , double &u , double &w , double &t , double &hr , double &ht ) {
  const_theta(z,hr,ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
  t = t + sample_cosine(x,z, 20.,xlen/2,2000.,2000.,2000.);
  t = t + sample_cosine(x,z,-20.,xlen/2,8000.,2000.,2000.);
}


//Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background density and potential temperature
void const_theta( double z , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p,exner,rt;
  //Establish balance first using Exner pressure
  t = theta0;                                  //Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                 //Pressure at z
  rt = pow((p / C0),(1. / gamm));             //rho*theta at z
  r = rt / t;                                  //Density at z
}


//Establish hydrstatic balance using constant Brunt-Vaisala frequency
//z is the input coordinate
//bv_freq0 is the constant Brunt-Vaisala frequency
//r and t are the output background density and potential temperature
void const_bvfreq( double z , double bv_freq0 , double &r , double &t ) {
  const double theta0 = 300.;  //Background potential temperature
  const double exner0 = 1.;    //Surface-level Exner pressure
  double       p, exner, rt;
  t = theta0 * exp( bv_freq0*bv_freq0 / grav * z );                                    //Pot temp at z
  exner = exner0 - grav*grav / (cp * bv_freq0*bv_freq0) * (t - theta0) / (t * theta0); //Exner pressure at z
  p = p0 * pow(exner,(cp/rd));                                                         //Pressure at z
  rt = pow((p / C0),(1. / gamm));                                                  //rho*theta at z
  r = rt / t;                                                                          //Density at z
}


//Sample from an ellipse of a specified center, radius, and amplitude at a specified location
//x and z are input coordinates
//amp,x0,z0,xrad,zrad are input amplitude, center, and radius of the ellipse
double sample_cosine( double x , double z , double amp , double x0 , double z0 , double xrad , double zrad ) {
  double dist;
  //Compute distance from bubble center
  dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) + ((z-z0)/zrad)*((z-z0)/zrad) ) * pi / 2.;
  //If the distance from bubble center is less than the radius, create a cos**2 profile
  if (dist <= pi / 2.) {
    return amp * pow(cos(dist),2.);
  } else {
    return 0.;
  }
}

void finalize() {
  free( state );
  free( state_tmp );
  free( flux );
  free( tend );
  free( cfd_dens_cell );
  free( cfd_dens_theta_cell );
  free( cfd_dens_int );
  free( cfd_dens_theta_int );
  free( cfd_pressure_int );
}


//Compute reduced quantities for error checking without resorting
void do_results( double &mass , double &te ) {
  mass = 0;
  te   = 0;
  for (int k=0; k<nnz; k++) {
    for (int i=0; i<nnx; i++) {
      int ind_r = POS_DENS*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
      int ind_u = POS_UMOM*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
      int ind_w = POS_WMOM*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
      int ind_t = POS_RHOT*(nnz+2*hs)*(nnx+2*hs) + (k+hs)*(nnx+2*hs) + i+hs;
      double r  =   state[ind_r] + cfd_dens_cell[hs+k];             // Density
      double u  =   state[ind_u] / r;                              // U-wind
      double w  =   state[ind_w] / r;                              // W-wind
      double th = ( state[ind_t] + cfd_dens_theta_cell[hs+k] ) / r; // Potential Temperature (theta)
      double p  = C0*pow(r*th,gamm);                               // Pressure
      double t  = th / pow(p0/p,rd/cp);                            // Temperature
      double ke = r*(u*u+w*w);                                     // Kinetic Energy
      double ie = r*cv*t;                                          // Internal Energy
      mass += r        *dx*dz; // Accumulate domain mass
      te   += (ke + ie)*dx*dz; // Accumulate domain total energy
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  ///////////////////////////////////////////////////////////////////////////////////////
  // BEGIN USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////
  //The x-direction length is twice as long as the z-direction length
  //So, you'll want to have nx_cfd be twice as large as nz_cfd
  nx_cfd = _NX;      //Number of total cells in the x-dirction
  nz_cfd = _NZ;       //Number of total cells in the z-dirction
  sim_time = _SIM_TIME;     //How many seconds to run the simulation
  output_freq = _OUT_FREQ;   //How frequently to output data to file (in seconds)
  config_spec = _IN_CONFIG;  //How to initialize the data
  ///////////////////////////////////////////////////////////////////////////////////////
  // END USER-CONFIGURABLE PARAMETERS
  ///////////////////////////////////////////////////////////////////////////////////////

  initialize( &argc , &argv );

  //Initial reductions for mass, kinetic energy, and total energy
  do_results(mass0,te0);

  ////////////////////////////////////////////////////
  // MAIN TIME STEP LOOP
  ////////////////////////////////////////////////////
  auto c_start = std::clock();
  while (etime < sim_time) {
    //If the time step leads to exceeding the simulation time, shorten it for the last step
    if (etime + dt > sim_time) { dt = sim_time - etime; }
    //Perform a single time step
    do_timestep(state,state_tmp,flux,tend,dt);
    //Update the elapsed time and output counter
    etime = etime + dt;
    output_counter = output_counter + dt;
    //If it's time for output, reset the counter, and do output
    if (output_counter >= output_freq) {
      output_counter = output_counter - output_freq;
      //Inform the user
      if (masterproc) { printf( "Elapsed Time: %lf / %lf\n", etime , sim_time ); }  
    }
  }
  auto c_end = std::clock();
  if (masterproc) {
    std::cout << "CPU Time: " << ( (double) (c_end-c_start) ) / CLOCKS_PER_SEC << " sec\n";
  }

  //Final reductions for mass, kinetic energy, and total energy
  do_results(mass,te);

  if (masterproc) {
    printf( "d_mass: %le\n" , (mass - mass0)/mass0 );
    printf( "d_te:   %le\n" , (te   - te0  )/te0   );
  }

  finalize();
}


