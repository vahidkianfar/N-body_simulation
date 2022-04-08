/*
 * A simple gravitational N-body simulation
 * Compile with a C compiler
 * For example, with gcc
 * gcc -o simulation simulation.c -std=c99 -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
static double omp_get_wtime()
{
  return (double)clock() / CLOCKS_PER_SEC;
}
#endif

/* Define data structures */
/* A 3D point. */
struct _p_Point {
   double x, y, z;
};

typedef struct _p_Point Point;

struct _p_Simulation {
  int dumpInterval;             /* interval for output */
  int N;                        /* number of particles */
  double endTime;               /* Final time for simulation */
  double dt;                    /* Timestep size */
  const char *output;           /* Output file, may be NULL */
  Point *positions;             /* Array of particle positions */
  Point *velocities;            /* Array of particle velocities */
  double *masses;               /* Array of particle masses */
};

typedef struct _p_Simulation *Simulation;

/* Set up simulation */
static void SimulationCreate(int argc, char **argv, Simulation *sim)
{
  Simulation s = calloc(1, sizeof(**sim));
  FILE *input = NULL;
  size_t nread;
  if (!s) {
    fprintf(stderr, "Unable to allocate space for simulation\n");
    exit(1);
  }
  ++argv;
  --argc;
  s->dt = strtod(argv[0], NULL);
  s->endTime = strtod(argv[1], NULL);

  if (argc >= 4) {
    s->output = argv[3];
    printf("%s\n", s->output);
    if (argc == 5) {
      s->dumpInterval = strtol(argv[4], NULL, 10);
    } else {
      s->dumpInterval = 1;
    }
  } else {
    s->output = NULL;
  }
  input = fopen(argv[2], "r");
  if (!input) {
    fprintf(stderr, "Unable to open input file '%s' for reading\n", argv[2]);
    exit(1);
  }
  nread = fread(&s->N, sizeof(s->N), 1, input);
  if (nread != 1) {
    fprintf(stderr, "Did not read number of particles from file\n");
    exit(1);
  }
  s->positions = calloc(s->N, sizeof(*s->positions));
  s->velocities = calloc(s->N, sizeof(*s->velocities));
  s->masses = calloc(s->N, sizeof(*s->masses));
  if (!s->positions) {
    fprintf(stderr, "Unable to allocate space for positions\n");
    exit(1);
  }
  if (!s->velocities) {
    fprintf(stderr, "Unable to allocate space for velocities\n");
    exit(1);
  }
  if (!s->masses) {
    fprintf(stderr, "Unable to allocate space for masses\n");
    exit(1);
  }
  nread = fread(s->positions, sizeof(*s->positions), s->N, input);
  if (nread != (size_t)s->N) {
    fprintf(stderr, "Unable to read positions from file\n");
    exit(1);
  }
  nread = fread(s->velocities, sizeof(*s->velocities), s->N, input);
  if (nread != (size_t)s->N) {
    fprintf(stderr, "Unable to read velocities from file\n");
    exit(1);
  }
  nread = fread(s->masses, sizeof(*s->masses), s->N, input);
  if (nread != (size_t)s->N) {
    fprintf(stderr, "Unable to read masses from file\n");
    exit(1);
  }
  *sim = s;
  
  fclose(input);
  return;
}

/* Destroy simulation*/
static void SimulationDestroy(Simulation *s)
{
  free((*s)->positions);
  free((*s)->velocities);
  free((*s)->masses);
  free(*s);
  *s = NULL;
}

/* Create a directory for output. */
static void CreateDirectory(const char *dir)
{
  int result = mkdir(dir, 0777);
  if (result && errno != EEXIST) {
    fprintf(stderr, "Unable to create '%s' directory\n", dir);
    exit(1);
  }
}

/* Write output for visualisation */
static void WritePVD(Simulation s, const char *dir, int nsteps)
{
  char pvdname[128];
  FILE *output = NULL;
  snprintf(pvdname, sizeof(pvdname), "%s/%s.pvd", dir, s->output);
  output = fopen(pvdname, "w");
  if (!output) {
    fprintf(stderr, "Unable to open '%s' for writing\n", pvdname);
    exit(1);
  }
  fprintf(output, "<?xml version=\"1.0\"?>\n");
  fprintf(output, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  fprintf(output, "<Collection>\n");
  for (int i = 0; i < nsteps; i++) {
    fprintf(output, "<DataSet timestep=\"%d\" group=\"\" part=\"0\" file=\"%s-%d.vtp\"/>\n", i, s->output, i);
  }
  fprintf(output, "</Collection>\n");
  fprintf(output, "</VTKFile>\n");
  fclose(output);
}

/* Write output for visualisation*/
static void WriteVTP(Simulation s, const char *dir, int step)
{
  char vtpname[128];
  FILE *output;
  snprintf(vtpname, sizeof(vtpname), "%s/%s-%d.vtp", dir, s->output, step);
  output = fopen(vtpname, "w");
  if (!output) {
    fprintf(stderr, "Unable to open '%s' for writing\n", vtpname);
    exit(1);
  }
  fprintf(output, "<?xml version=\"1.0\"?>\n");
  fprintf(output, "<VTKFile type=\"PolyData\" >\n");
  fprintf(output, "<PolyData>\n");
  fprintf(output, "<Piece NumberOfPoints=\"%d\">\n", s->N);
  fprintf(output, "<Points>\n");
  fprintf(output, "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (int i = 0; i < s->N; i++) {
    Point xi = s->positions[i];
    fprintf(output, "%g %g %g ", xi.x, xi.y, xi.z);
  }
  fprintf(output, "</DataArray>\n");
  fprintf(output, "</Points>\n");
  fprintf(output, "</Piece>\n");
  fprintf(output, "</PolyData>\n");
  fprintf(output, "</VTKFile>\n");
  fclose(output);
}

/**
 * This function takes as input a Simulation object (containing
 * particles) and an array of Points representing the accelerations.
 * These accelerations are updated in place.
 */
static void CalculateAccelerations(Simulation s, Point *accelerations,int nthreads)
{
  int i,j=0;  
  /* Calculate all the pairwise accelerations */
  #pragma omp parallel private(i,j) num_threads(nthreads)
  {
  #pragma omp for schedule(guided,2) nowait
  for (i = 0; i < s->N; i++) {
    /* Current point */
    const Point xi = s->positions[i];
    /* Pointer to the current point's acceleration */
    Point *ai = &accelerations[i];
    /* Initialise to zero. */
    ai->x = 0;
    ai->y = 0;
    ai->z = 0;
    /* Pairwise interaction with all other points. */
   
    
    for (j = 0; j < s->N; j++) {
      const Point xj = s->positions[j];
      Point diff;
      if (i == j) {
        /* skip self-interaction */
        continue;
      }
    
      //#pragma omp task
      diff.x = xj.x - xi.x;
      diff.y = xj.y - xi.y;
      diff.z = xj.z - xi.z;
      
      /* Squared distance between xi and xj. */
      
      const double dist = sqrt(pow(diff.x, 2)
                               + pow(diff.y, 2)
                               + pow(diff.z, 2));
      const double factor = s->masses[j] / pow(dist, 3);
      
      /* Force is mass[i] * mass[j] * (xj - xi) / ||xj - xi||^3 */
      /* Acceleration[i] is force/mass[i]. */
      /* Update the acceleration with the interaction with this point. */
      
      ai->x += diff.x * factor;
      ai->y += diff.y * factor;
      ai->z += diff.z * factor;
      
    }
    
  }
  
}
}

static int SimulationRun(Simulation s,int nthreads)
{
  const char *output_directory = "results";
  /* Space to hold all accelerations. */
  Point *accelerations = calloc(s->N, sizeof(*accelerations));

  double t = 0;
  int step = 0;
  int counter = 0;
  int i=0;
 
  /* Write initial positions if requested. */
  if (s->output) {
    CreateDirectory(output_directory);
    WriteVTP(s, output_directory, step++);
  }
  /* Velocity verlet algorithm
   * https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
   * This is an energy-conserving explicit scheme.
   */
  /* Initial accelerations */
  
  CalculateAccelerations(s, accelerations,nthreads);
  
  while (t < s->endTime) {
    /* Update every point. */
    #pragma omp parallel num_threads(nthreads)
    {
    #pragma omp for schedule(guided,2) private(i) nowait
    for (i = 0; i < s->N; i++) {
    Point *xi = &s->positions[i];
    Point *vi = &s->velocities[i];
    Point *ai = &accelerations[i];
    /* Update positions */
    xi->x = xi->x + vi->x * s->dt + 0.5*ai->x * pow(s->dt, 2);
    xi->y = xi->y + vi->y * s->dt + 0.5*ai->y * pow(s->dt, 2);
    xi->z = xi->z + vi->z * s->dt + 0.5*ai->z * pow(s->dt, 2);
    /* Half the velocity step (with the old accelerations) */
    vi->x = vi->x + 0.5 * ai->x * s->dt;
    vi->y = vi->y + 0.5 * ai->y * s->dt;
    vi->z = vi->z + 0.5 * ai->z * s->dt;
      }
    //}
    }  
    /* Compute the new accelerations */
    //#pragma omp single
    CalculateAccelerations(s, accelerations,nthreads);
    
   // }
     #pragma omp parallel num_threads(nthreads)
     {
    #pragma omp for schedule(guided,2) private(i) nowait
    for (i = 0; i < s->N; i++) {
      Point *vi = &s->velocities[i];
      Point *ai = &accelerations[i];
      
      /* Finish the velocity update */
      //#pragma omp task
      vi->x = vi->x + 0.5 * ai->x * s->dt;
      vi->y = vi->y + 0.5 * ai->y * s->dt;
      vi->z = vi->z + 0.5 * ai->z * s->dt;
    }
     }
    // }
  
    
    /* Update simulation time and counter for output. */
    t += s->dt;
    counter++;
    
  // free(accelerations);
    /* Write output for visualisation if requested. */
    if (s->output && counter % s->dumpInterval == 0) {
      WriteVTP(s, output_directory, step++);
    }
  }
  if (s->output) {
    /* Write "summary" PVD file. */
    WritePVD(s, output_directory, step);
  }
  /* Clean up memory. */
  
  free(accelerations);
  accelerations = NULL;
  
  return counter;
}

/* Print usage information. */
static void usage(const char *basename)
{
  printf("Usage: %s DT END-TIME INPUT-FILE [OUTPUT-FILE [SNAPSHOT]]\n", basename);
  printf("    DT          Time step size\n");
  printf("    END-TIME    Final simulated time\n");
  printf("    INPUT-FILE  Name of file containing initial positions of particles\n");
  printf("    OUTPUT-FILE Name of output file for visualisation of outputs (in results subdirectory)\n");
  printf("                (optional, if not provided, no visualisation is produced)\n");
  printf("    SNAPSHOT    Frequency of output (e.g. if given as 10 then output is\n");
  printf("                produced every 10 timesteps). Optional, defaults to 1.\n");
  return;
}

/* Print some statistics about the simulation.
 * Useful for debugging. For example, we expect conservation of total
 * system momentum (up to machine precision). */
static void SimulationPrintStats(Simulation s)
{
  int N = s->N < 3 ? s->N : 3;
  printf("Simulation with %d particles\n", s->N);
  printf("Timestep is %g, end time is %g\n", s->dt, s->endTime);
  if (s->output) {
    printf("Producing visualisation output in %s.pvd\n", s->output);
  } else {
    printf("Not producing visualisation output\n");
  }
  Point momentum;
  momentum.x = 0;
  momentum.y = 0;
  momentum.z = 0;
  for (int i = 0; i < s->N; i++) {
    const double mass = s->masses[i];
    const Point vi = s->velocities[i];
    momentum.x += vi.x * mass;
    momentum.y += vi.y * mass;
    momentum.z += vi.z * mass;
  }
  printf("Total system momentum is [%g, %g, %g]\n", momentum.x, momentum.y, momentum.z);

  printf("Details on first %d particles\n", N);
  for (int i = 0; i < N; i++) {
    const Point xi = s->positions[i];
    const Point vi = s->velocities[i];
    const double mass = s->masses[i];
    printf("x = [%g, %g, %g]; v = [%g, %g, %g]; m = %g\n", xi.x, xi.y, xi.z, vi.x, vi.y, vi.z, mass);
  }
}
/*
 * Main routine.
 */
int main(int argc, char** argv) {
  Simulation simulation;
  double start, end;
  double duration;
  int nstep;
  int nthreads=56;
  
  if (argc < 4 || argc > 6) {
    usage(basename(argv[0]));
    return 1;
  }
  SimulationCreate(argc, argv, &simulation);
  printf("Initial statistics\n");
  SimulationPrintStats(simulation);
  printf("Starting simulation\n");
  start = omp_get_wtime();
  nstep = SimulationRun(simulation,nthreads);
  end = omp_get_wtime();
  printf("Finished simulation\n");
  printf("Final statistics\n");
  SimulationPrintStats(simulation);
  duration = end - start;
  printf("Basic timing results\n");
  printf("Total time: %gs\n", duration);
  printf("Time per step: %gs\n", duration/nstep);
  printf("Time per particle per step: %gs\n", duration/(nstep * simulation->N));
  SimulationDestroy(&simulation);
  
  return 0;
}
