!> Shared precision kinds for fMagCalc.
!> dp = float64 (matches NumPy float64), cp = complex128.
module magcalc_kinds
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   private
   public :: dp, cp

   integer, parameter :: dp = real64
   integer, parameter :: cp = real64   !< kind for the real/imag parts of complex(kind=cp)
end module magcalc_kinds
