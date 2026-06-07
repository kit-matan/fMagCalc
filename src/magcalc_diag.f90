!> Non-Hermitian eigen-decomposition wrapper (LAPACK zgeev) and the
!> dispersion-only path. Mirrors process_calc_disp in pyMagCalc/magcalc/numerical.py.
!>
!> STATUS: stub. eig_zgeev is implemented as the first concrete building block
!> (M1); the dispersion sort/selection is sketched and must be validated against
!> process_calc_disp before use.
module magcalc_diag
   use magcalc_kinds, only: dp, cp
   implicit none
   private
   public :: eig_zgeev

contains

   !> Compute eigenvalues (and optionally right eigenvectors) of a general
   !> complex matrix A (n x n) via LAPACK zgeev. A is not modified.
   subroutine eig_zgeev(n, a, w, vr, info)
      integer,            intent(in)  :: n
      complex(kind=cp),   intent(in)  :: a(n, n)
      complex(kind=cp),   intent(out) :: w(n)        !< eigenvalues
      complex(kind=cp),   intent(out) :: vr(n, n)    !< right eigenvectors (columns)
      integer,            intent(out) :: info

      complex(kind=cp), allocatable :: acopy(:, :), work(:), vl(:, :)
      complex(kind=cp)              :: work_query(1)
      real(kind=dp),    allocatable :: rwork(:)
      integer :: lwork

      allocate (acopy(n, n)); acopy = a
      allocate (vl(1, 1))
      allocate (rwork(2 * n))

      ! Workspace query.
      call zgeev('N', 'V', n, acopy, n, w, vl, 1, vr, n, work_query, -1, rwork, info)
      lwork = max(1, int(real(work_query(1))))
      allocate (work(lwork))

      call zgeev('N', 'V', n, acopy, n, w, vl, 1, vr, n, work, lwork, rwork, info)

      deallocate (acopy, vl, work, rwork)
   end subroutine eig_zgeev

end module magcalc_diag
