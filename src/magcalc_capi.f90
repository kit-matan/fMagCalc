!> C-interoperable entry points for the in-process (ctypes) path. These wrap the
!> OpenMP drivers so Python can call them directly on NumPy array pointers — no
!> file round-trip. Arrays are column-major (Fortran); the Python side arranges
!> memory accordingly (see python/fmagcalc/_capi.py).
!>
!> Chosen over f2py because the CMake build already links LAPACK (Accelerate)
!> and OpenMP cleanly; a bind(C) shared library reuses that with no extra
!> build-system friction.
module magcalc_capi
   use, intrinsic :: iso_c_binding, only: c_int, c_double
   use magcalc_kinds,  only: dp, cp
   use magcalc_driver, only: run_dispersion
   use magcalc_sqw,    only: run_sqw, run_sqw_model
   implicit none
   private
   public :: magcalc_disp_c, magcalc_sqw_c, magcalc_sqw_model_c

contains

   subroutine magcalc_disp_c(n, nq, h_plus, energies, info) &
         bind(C, name="magcalc_disp_c")
      integer(c_int),   value       :: n, nq
      complex(kind=cp), intent(in)  :: h_plus(2*n, 2*n, nq)
      real(kind=dp),    intent(out) :: energies(n, nq)
      integer(c_int),   intent(out) :: info(nq)
      integer :: iinfo(nq)
      call run_dispersion(n, nq, h_plus, energies, iinfo)
      info = int(iinfo, c_int)
   end subroutine magcalc_disp_c

   subroutine magcalc_sqw_c(n, nq, s, qgrid, hplus, hminus, ud, ff, &
                            energies, intensities, info, evals) &
         bind(C, name="magcalc_sqw_c")
      integer(c_int),   value       :: n, nq
      real(c_double),   value       :: s
      real(kind=dp),    intent(in)  :: qgrid(3, nq), ff(n, nq)
      complex(kind=cp), intent(in)  :: hplus(2*n, 2*n, nq), hminus(2*n, 2*n, nq)
      complex(kind=cp), intent(in)  :: ud(3*n, 3*n)
      real(kind=dp),    intent(out) :: energies(n, nq), intensities(n, nq)
      integer(c_int),   intent(out) :: info(nq)
      complex(kind=cp), intent(out) :: evals(2*n, nq)

      complex(kind=cp), allocatable :: K(:, :, :), Kd(:, :, :)
      integer :: iinfo(nq)

      ! K/Kd are needed by the kernel but are gauge-dependent and not returned
      ! to Python; allocate scratch and discard.
      allocate(K(3*n, 2*n, nq), Kd(3*n, 2*n, nq))
      call run_sqw(n, nq, real(s, dp), qgrid, hplus, hminus, ud, ff, &
                   energies, intensities, iinfo, K, Kd, evals)
      info = int(iinfo, c_int)
      deallocate(K, Kd)
   end subroutine magcalc_sqw_c

   !> Phase-2: build H(q) from the bond model inside the Fortran loop. Only Mb
   !> (2N,2N,nb) crosses the boundary, not a per-q H stack.
   subroutine magcalc_sqw_model_c(n, nq, nb, s, qgrid, dvec, Mb, ud, ff, &
                                  energies, intensities, info, evals) &
         bind(C, name="magcalc_sqw_model_c")
      integer(c_int),   value       :: n, nq, nb
      real(c_double),   value       :: s
      real(kind=dp),    intent(in)  :: qgrid(3, nq), dvec(3, nb), ff(n, nq)
      complex(kind=cp), intent(in)  :: Mb(2*n, 2*n, nb), ud(3*n, 3*n)
      real(kind=dp),    intent(out) :: energies(n, nq), intensities(n, nq)
      integer(c_int),   intent(out) :: info(nq)
      complex(kind=cp), intent(out) :: evals(2*n, nq)
      integer :: iinfo(nq)
      call run_sqw_model(n, nq, nb, real(s, dp), qgrid, dvec, Mb, ud, ff, &
                         energies, intensities, iinfo, evals)
      info = int(iinfo, c_int)
   end subroutine magcalc_sqw_model_c

end module magcalc_capi
