!> S(Q,w) per-q intensity contraction + OpenMP grid driver. Mirrors the second
!> half of process_calc_Sqw in pyMagCalc/magcalc/numerical.py:
!>   energies   = Re(eigvals[0:N])                 (positive branch)
!>   A[a,mode]  = sum_i ff_i * K[3i+a, mode]
!>   B[b,mode]  = sum_j ff_j * Kd[3j+b, N+mode]
!>   I[mode]    = Re( sum_{a,b} P[a,b] A[a,mode] B[b,mode] ),  clamp(>=0)
!> with P = I3 - qhat (x) qhat  (P = I3 when |q| ~ 0).
module magcalc_sqw
   use magcalc_kinds, only: dp, cp
   use magcalc_kkd,   only: kkd_matrix
   implicit none
   private
   public :: run_sqw

   real(kind=dp), parameter :: Q_ZERO_THRESHOLD = 1.0e-10_dp

contains

   subroutine run_sqw(n, nq, s, qgrid, hplus, hminus, ud, ff, &
                      energies, intensities, info, Kout, Kdout, evout)
      integer,          intent(in)  :: n, nq
      real(kind=dp),    intent(in)  :: s
      real(kind=dp),    intent(in)  :: qgrid(3, nq)
      complex(kind=cp), intent(in)  :: hplus(2*n, 2*n, nq)
      complex(kind=cp), intent(in)  :: hminus(2*n, 2*n, nq)
      complex(kind=cp), intent(in)  :: ud(3*n, 3*n)
      real(kind=dp),    intent(in)  :: ff(n, nq)
      real(kind=dp),    intent(out) :: energies(n, nq)
      real(kind=dp),    intent(out) :: intensities(n, nq)
      integer,          intent(out) :: info(nq)
      complex(kind=cp), intent(out) :: Kout(3*n, 2*n, nq)   !< debug/validation
      complex(kind=cp), intent(out) :: Kdout(3*n, 2*n, nq)  !< debug/validation
      complex(kind=cp), intent(out) :: evout(2*n, nq)       !< debug/validation

      integer          :: iq, ierr
      complex(kind=cp) :: k(3*n, 2*n), kd(3*n, 2*n), evals(2*n)

      !$omp parallel do default(shared) private(iq, k, kd, evals, ierr) schedule(dynamic)
      do iq = 1, nq
         call kkd_matrix(s, n, hplus(:, :, iq), hminus(:, :, iq), ud, k, kd, evals, ierr)
         info(iq) = ierr
         Kout(:, :, iq)  = k
         Kdout(:, :, iq) = kd
         evout(:, iq)    = evals
         if (ierr /= 0) then
            energies(:, iq)    = ieee_nan()
            intensities(:, iq) = ieee_nan()
            cycle
         end if
         call sqw_intensity(n, qgrid(:, iq), ff(:, iq), k, kd, evals, &
                            energies(:, iq), intensities(:, iq))
      end do
      !$omp end parallel do
   contains
      real(kind=dp) function ieee_nan() result(x)
         use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
         x = ieee_value(0.0_dp, ieee_quiet_nan)
      end function ieee_nan
   end subroutine run_sqw

   subroutine sqw_intensity(n, qvec, ff, k, kd, evals, energies, intensities)
      integer,          intent(in)  :: n
      real(kind=dp),    intent(in)  :: qvec(3), ff(n)
      complex(kind=cp), intent(in)  :: k(3*n, 2*n), kd(3*n, 2*n), evals(2*n)
      real(kind=dp),    intent(out) :: energies(n), intensities(n)

      integer          :: i, a, b, mode
      complex(kind=cp) :: amat(3, n), bmat(3, n), acc
      real(kind=dp)    :: qn2, pol(3, 3), qhat(3)

      energies = real(evals(1:n), dp)

      ! A[a,mode] = sum_i ff_i K[3(i-1)+a, mode]; B[b,mode] = sum_j ff_j Kd[3(j-1)+b, N+mode]
      amat = (0.0_dp, 0.0_dp); bmat = (0.0_dp, 0.0_dp)
      do mode = 1, n
         do i = 1, n
            do a = 1, 3
               amat(a, mode) = amat(a, mode) + ff(i) * k(3*(i-1)+a, mode)
               bmat(a, mode) = bmat(a, mode) + ff(i) * kd(3*(i-1)+a, n + mode)
            end do
         end do
      end do

      qn2 = dot_product(qvec, qvec)
      pol = 0.0_dp
      do a = 1, 3
         pol(a, a) = 1.0_dp
      end do
      if (qn2 >= Q_ZERO_THRESHOLD) then
         qhat = qvec / sqrt(qn2)
         do a = 1, 3
            do b = 1, 3
               pol(a, b) = pol(a, b) - qhat(a) * qhat(b)
            end do
         end do
      end if

      do mode = 1, n
         acc = (0.0_dp, 0.0_dp)
         do a = 1, 3
            do b = 1, 3
               acc = acc + pol(a, b) * amat(a, mode) * bmat(b, mode)
            end do
         end do
         intensities(mode) = real(acc, dp)
         if (intensities(mode) < 0.0_dp) intensities(mode) = 0.0_dp
      end do
   end subroutine sqw_intensity

end module magcalc_sqw
