#include <slepceps.h>

int main(int argc, char **argv)
{
  Mat      A;
  EPS      eps;
  PetscInt nconv;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetValue(A, 0, 0, 1.0, INSERT_VALUES));
  PetscCall(MatSetValue(A, 0, 1, 2.0, INSERT_VALUES));
  PetscCall(MatSetValue(A, 1, 1, 3.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
  PetscCall(EPSSetOperators(eps, A, NULL));
  PetscCall(EPSSetProblemType(eps, EPS_NHEP));
  PetscCall(EPSSetType(eps, EPSKRYLOVSCHUR));
  PetscCall(EPSSetDimensions(eps, 1, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(EPSSolve(eps));
  PetscCall(EPSGetConverged(eps, &nconv));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "native_c_nhep_nconv=%" PetscInt_FMT "\n", nconv));

  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}
