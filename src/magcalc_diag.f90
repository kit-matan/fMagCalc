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
   public :: eig_zgeev, argsort_complex, argsort_abs, qr_orthonormal

contains

   !> Stable ascending argsort of a complex array by (real, then imag) — matches
   !> numpy's complex sort order. Returns 1-based permutation in idx.
   subroutine argsort_complex(m, z, idx)
      integer,          intent(in)  :: m
      complex(kind=cp), intent(in)  :: z(m)
      integer,          intent(out) :: idx(m)
      integer :: i, j, key
      do i = 1, m
         idx(i) = i
      end do
      do i = 2, m
         key = idx(i)
         j = i - 1
         do while (j >= 1)
            if (.not. less_complex(z(key), z(idx(j)))) exit
            idx(j+1) = idx(j)
            j = j - 1
         end do
         idx(j+1) = key
      end do
   end subroutine argsort_complex

   pure logical function less_complex(a, b) result(lt)
      complex(kind=cp), intent(in) :: a, b
      if (real(a, dp) < real(b, dp)) then
         lt = .true.
      else if (real(a, dp) > real(b, dp)) then
         lt = .false.
      else
         lt = aimag(a) < aimag(b)
      end if
   end function less_complex

   !> Stable ascending argsort of a complex array by magnitude |z|.
   subroutine argsort_abs(m, z, idx)
      integer,          intent(in)  :: m
      complex(kind=cp), intent(in)  :: z(m)
      integer,          intent(out) :: idx(m)
      real(kind=dp) :: mag(m), keyval
      integer       :: i, j, key
      do i = 1, m
         idx(i) = i
         mag(i) = abs(z(i))
      end do
      do i = 2, m
         key = idx(i); keyval = mag(key)
         j = i - 1
         do while (j >= 1)
            if (mag(idx(j)) <= keyval) exit
            idx(j+1) = idx(j)
            j = j - 1
         end do
         idx(j+1) = key
      end do
   end subroutine argsort_abs

   !> Economy QR orthonormalization (columns of a become orthonormal, spanning
   !> the same space). Uses the same LAPACK path (zgeqrf + zungqr) as numpy.qr,
   !> so the resulting Q matches numpy to round-off. a is overwritten with Q.
   subroutine qr_orthonormal(nrow, ncol, a, info)
      integer,          intent(in)    :: nrow, ncol
      complex(kind=cp), intent(inout) :: a(nrow, ncol)
      integer,          intent(out)   :: info
      complex(kind=cp), allocatable :: tau(:), work(:)
      complex(kind=cp)              :: wq(1)
      integer :: lwork, k

      k = min(nrow, ncol)
      allocate(tau(k))
      call zgeqrf(nrow, ncol, a, nrow, tau, wq, -1, info)
      lwork = max(1, int(real(wq(1))))
      allocate(work(lwork))
      call zgeqrf(nrow, ncol, a, nrow, tau, work, lwork, info)
      if (info /= 0) then
         deallocate(tau, work); return
      end if
      deallocate(work)
      call zungqr(nrow, ncol, k, a, nrow, tau, wq, -1, info)
      lwork = max(1, int(real(wq(1))))
      allocate(work(lwork))
      call zungqr(nrow, ncol, k, a, nrow, tau, work, lwork, info)
      deallocate(tau, work)
   end subroutine qr_orthonormal

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
