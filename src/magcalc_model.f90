!> Phase-2 model build: reconstruct the dynamical matrix H(q) from its Fourier
!> (bond) decomposition  H(q) = sum_b M_b * exp(i q.d_b)  so the q-loop builds
!> H itself — no (Nq,2N,2N) matrix stack crosses the language boundary, only the
!> per-bond coefficient matrices M_b (built once). See PLAN.md §2 / docs/INTERFACE.md.
module magcalc_model
   use magcalc_kinds, only: dp, cp
   implicit none
   private
   public :: build_h

   complex(kind=cp), parameter :: II = (0.0_dp, 1.0_dp)

contains

   !> H(2N,2N) = sum_b Mb(:,:,b) * exp(i * dot(q, dvec(:,b)))
   subroutine build_h(n, nb, dvec, Mb, q, H)
      integer,          intent(in)  :: n, nb
      real(kind=dp),    intent(in)  :: dvec(3, nb)
      complex(kind=cp), intent(in)  :: Mb(2*n, 2*n, nb)
      real(kind=dp),    intent(in)  :: q(3)
      complex(kind=cp), intent(out) :: H(2*n, 2*n)
      integer          :: b
      complex(kind=cp) :: ph

      H = (0.0_dp, 0.0_dp)
      do b = 1, nb
         ph = exp(II * (q(1)*dvec(1,b) + q(2)*dvec(2,b) + q(3)*dvec(3,b)))
         H = H + Mb(:, :, b) * ph
      end do
   end subroutine build_h

end module magcalc_model
