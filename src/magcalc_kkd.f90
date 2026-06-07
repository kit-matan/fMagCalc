!> Bogoliubov pipeline for a single q-point: the Fortran counterpart of
!> KKdMatrix in pyMagCalc/magcalc/linalg.py.
!>
!> STATUS: stub. This is the most subtle module to port. Implement it last and
!> validate step-by-step against linalg.py:
!>   _diagonalize_and_sort  -> two-stage sort (split at N, re-sort |lambda| block)
!>   _apply_gram_schmidt    -> degenerate-block QR (zgeqrf/zungqr)
!>   _calculate_alpha_matrix-> G-metric pseudo-norm, alpha = sqrt(G_ii / N_ii)
!>   _match_and_reorder_minus_q -> projection matching + phase factor
!>   _calculate_K_Kd        -> sqrt(S/2) * Ud * Udd * inv_T  (zgemm)
module magcalc_kkd
   use magcalc_kinds, only: dp, cp
   implicit none
   private
   public :: kkd_matrix

   real(kind=dp), parameter :: DEGENERACY_THRESHOLD           = 1.0e-8_dp
   real(kind=dp), parameter :: ZERO_MATRIX_ELEMENT_THRESHOLD  = 1.0e-6_dp
   real(kind=dp), parameter :: EIGENVECTOR_MATCHING_THRESHOLD = 1.0e-4_dp

contains

   !> Returns K (3N x 2N), Kd (3N x 2N) and the +q eigenvalues (2N).
   subroutine kkd_matrix(s, n, hplus, hminus, ud, k, kd, evals, info)
      real(kind=dp),    intent(in)  :: s
      integer,          intent(in)  :: n
      complex(kind=cp), intent(in)  :: hplus(2*n, 2*n)
      complex(kind=cp), intent(in)  :: hminus(2*n, 2*n)
      complex(kind=cp), intent(in)  :: ud(3*n, 3*n)
      complex(kind=cp), intent(out) :: k(3*n, 2*n)
      complex(kind=cp), intent(out) :: kd(3*n, 2*n)
      complex(kind=cp), intent(out) :: evals(2*n)
      integer,          intent(out) :: info

      ! TODO(M2): implement the full pipeline. For now flag "not implemented".
      k = (0.0_dp, 0.0_dp)
      kd = (0.0_dp, 0.0_dp)
      evals = (0.0_dp, 0.0_dp)
      info = -99
   end subroutine kkd_matrix

end module magcalc_kkd
