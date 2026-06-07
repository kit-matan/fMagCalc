!> OpenMP q-loop drivers. These are the entry points the Python wrapper calls
!> (via f2py) for the Phase-1 matrix-stack handoff. See docs/INTERFACE.md.
!>
!> STATUS: dispersion driver is wired to eig_zgeev as the first end-to-end
!> slice (M1); the S(Q,w) driver is a stub pending magcalc_kkd (M2).
module magcalc_driver
   use magcalc_kinds, only: dp, cp
   use magcalc_diag,  only: eig_zgeev
   implicit none
   private
   public :: run_dispersion

contains

   !> Dispersion over a q-grid. h_plus is (2N,2N,Nq) so each slab h_plus(:,:,iq)
   !> is contiguous column-major. energies is (N, Nq).
   subroutine run_dispersion(n, nq, h_plus, energies, info)
      integer,          intent(in)  :: n, nq
      complex(kind=cp), intent(in)  :: h_plus(2*n, 2*n, nq)
      real(kind=dp),    intent(out) :: energies(n, nq)
      integer,          intent(out) :: info(nq)

      integer          :: iq, m, ierr
      complex(kind=cp) :: w(2*n), vr(2*n, 2*n)
      real(kind=dp)    :: re(2*n)

      m = 2 * n
      !$omp parallel do default(shared) private(iq, w, vr, re, ierr) schedule(dynamic)
      do iq = 1, nq
         call eig_zgeev(m, h_plus(:, :, iq), w, vr, ierr)
         info(iq) = ierr
         if (ierr /= 0) then
            energies(:, iq) = ieee_nan()
            cycle
         end if
         ! Sort real parts ascending, take the upper N (the magnon branch).
         re = real(w, kind=dp)
         call sort_real_ascending(m, re)
         energies(:, iq) = re(n+1:m)
      end do
      !$omp end parallel do
   contains
      pure real(kind=dp) function ieee_nan() result(x)
         use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
         x = ieee_value(0.0_dp, ieee_quiet_nan)
      end function ieee_nan
   end subroutine run_dispersion

   !> Simple ascending insertion sort (m is small: 2N). Replace with a heap/
   !> quicksort if 2N grows large.
   subroutine sort_real_ascending(m, x)
      integer,       intent(in)    :: m
      real(kind=dp), intent(inout) :: x(m)
      integer       :: i, j
      real(kind=dp) :: key
      do i = 2, m
         key = x(i)
         j = i - 1
         do while (j >= 1)
            if (x(j) <= key) exit
            x(j+1) = x(j)
            j = j - 1
         end do
         x(j+1) = key
      end do
   end subroutine sort_real_ascending

end module magcalc_driver
