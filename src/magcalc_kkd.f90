!> Bogoliubov pipeline for a single q-point: the Fortran counterpart of
!> KKdMatrix in pyMagCalc/magcalc/linalg.py. Ported step-by-step to mirror the
!> Python helpers; validated against exported K/Kd/eigvals intermediates.
!>
!>   diag_and_sort   <- _diagonalize_and_sort  (two-stage sort)
!>   apply_gs        <- _apply_gram_schmidt    (degenerate-block QR)
!>   calc_alpha      <- _calculate_alpha_matrix (G-metric pseudo-norm)
!>   match_reorder   <- _match_and_reorder_minus_q (projection match + phase)
!>   K/Kd build      <- _calculate_K_Kd        (sqrt(S/2) * Ud * Udd * inv_T)
module magcalc_kkd
   use magcalc_kinds, only: dp, cp
   use magcalc_diag,  only: eig_zgeev, argsort_complex, argsort_abs, qr_orthonormal
   implicit none
   private
   public :: kkd_matrix

   real(kind=dp), parameter :: DEGENERACY_THRESHOLD           = 1.0e-8_dp
   real(kind=dp), parameter :: ZERO_MATRIX_ELEMENT_THRESHOLD  = 1.0e-6_dp
   real(kind=dp), parameter :: ALPHA_ZERO_NORM_WARN           = 1.0e-14_dp
   real(kind=dp), parameter :: EIGENVECTOR_MATCHING_THRESHOLD = 1.0e-4_dp
   complex(kind=cp), parameter :: II = (0.0_dp, 1.0_dp)

contains

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

      integer          :: n2, n3, i, j
      real(kind=dp)    :: gdiag(2*n), pref
      complex(kind=cp) :: evp(2*n), evm(2*n)
      complex(kind=cp) :: Vp(2*n, 2*n), Vm(2*n, 2*n)
      complex(kind=cp) :: alphap(2*n), alpham(2*n)
      complex(kind=cp) :: Vm_re(2*n, 2*n), evm_re(2*n), alpham_re(2*n)
      complex(kind=cp) :: invTp(2*n, 2*n), invTm(2*n, 2*n)
      complex(kind=cp) :: Udd(3*n, 2*n)
      integer          :: ierr

      n2 = 2 * n; n3 = 3 * n
      info = 0
      do i = 1, n
         gdiag(i)     =  1.0_dp
         gdiag(i + n) = -1.0_dp
      end do

      ! 1. diagonalize + sort (+q),  2. GS,  3. alpha
      call diag_and_sort(n, hplus, evp, Vp, ierr)
      if (ierr /= 0) then; info = -1; return; end if
      call apply_gs(n2, evp, Vp)
      call calc_alpha(n2, Vp, gdiag, alphap)

      ! 4. diagonalize + sort (-q),  5. GS,  6. alpha
      call diag_and_sort(n, hminus, evm, Vm, ierr)
      if (ierr /= 0) then; info = -2; return; end if
      call apply_gs(n2, evm, Vm)
      call calc_alpha(n2, Vm, gdiag, alpham)

      ! 7. match & reorder -q to +q
      call match_reorder(n, Vp, alphap, Vm, evm, alpham, Vm_re, evm_re, alpham_re)

      ! 8. inverse transforms (alpha is diagonal: scale columns)
      do j = 1, n2
         invTp(:, j) = Vp(:, j)    * alphap(j)
         invTm(:, j) = Vm_re(:, j) * alpham_re(j)
      end do

      ! 9. K and Kd
      Udd = (0.0_dp, 0.0_dp)
      do i = 1, n
         ! rows are 1-based: python 3*i -> 3*(i-1)+1 ; 3*i+1 -> 3*(i-1)+2
         Udd(3*(i-1)+1, i)     = (1.0_dp, 0.0_dp)
         Udd(3*(i-1)+1, i + n) = (1.0_dp, 0.0_dp)
         Udd(3*(i-1)+2, i)     = -II    ! 1/I = -i
         Udd(3*(i-1)+2, i + n) =  II    ! -1/I = +i
      end do
      pref = sqrt(s / 2.0_dp)
      k  = pref * matmul(ud, matmul(Udd, invTp))
      kd = pref * matmul(ud, matmul(Udd, invTm))
      where (abs(k)  < ZERO_MATRIX_ELEMENT_THRESHOLD) k  = (0.0_dp, 0.0_dp)
      where (abs(kd) < ZERO_MATRIX_ELEMENT_THRESHOLD) kd = (0.0_dp, 0.0_dp)

      evals = evp
   end subroutine kkd_matrix

   !> _diagonalize_and_sort: zgeev, then order [positive ascending | negative by
   !> ascending |.|]. Columns of evecs follow the same permutation.
   subroutine diag_and_sort(n, Hmat, evals_sorted, evecs_sorted, info)
      integer,          intent(in)  :: n
      complex(kind=cp), intent(in)  :: Hmat(2*n, 2*n)
      complex(kind=cp), intent(out) :: evals_sorted(2*n)
      complex(kind=cp), intent(out) :: evecs_sorted(2*n, 2*n)
      integer,          intent(out) :: info

      integer          :: n2, i
      complex(kind=cp) :: w(2*n), vr(2*n, 2*n)
      integer          :: srt(2*n), srt_neg(n)
      complex(kind=cp) :: ev_neg(n)
      complex(kind=cp) :: vecs_top(2*n, n), vecs_bot(2*n, n)
      complex(kind=cp) :: ev_top(n)

      n2 = 2 * n
      call eig_zgeev(n2, Hmat, w, vr, info)
      if (info /= 0) return

      call argsort_complex(n2, w, srt)
      ! upper half (indices n+1..2n of ascending sort) = positive branch
      do i = 1, n
         ev_top(i)       = w(srt(n + i))
         vecs_top(:, i)  = vr(:, srt(n + i))
         ev_neg(i)       = w(srt(i))            ! lower half (negative)
         vecs_bot(:, i)  = vr(:, srt(i))
      end do
      ! re-sort the negative block by |.| ascending
      call argsort_abs(n, ev_neg, srt_neg)
      do i = 1, n
         evals_sorted(i)        = ev_top(i)
         evecs_sorted(:, i)     = vecs_top(:, i)
         evals_sorted(n + i)    = ev_neg(srt_neg(i))
         evecs_sorted(:, n + i) = vecs_bot(:, srt_neg(i))
      end do
   end subroutine diag_and_sort

   !> _apply_gram_schmidt: orthonormalize each block of consecutive eigenvectors
   !> whose eigenvalues are within DEGENERACY_THRESHOLD.
   subroutine apply_gs(m, evals, evecs)
      integer,          intent(in)    :: m
      complex(kind=cp), intent(in)    :: evals(m)
      complex(kind=cp), intent(inout) :: evecs(m, m)
      integer :: i, deg, s0, e0, blk, ierr
      complex(kind=cp), allocatable :: block(:, :)

      deg = 0
      do i = 2, m
         if (abs(evals(i) - evals(i-1)) < DEGENERACY_THRESHOLD) then
            deg = deg + 1
         else if (deg > 0) then
            s0 = i - deg - 1      ! 1-based start of block
            e0 = i - 1            ! 1-based end of block
            blk = e0 - s0 + 1
            allocate(block(m, blk)); block = evecs(:, s0:e0)
            call qr_orthonormal(m, blk, block, ierr)
            if (ierr == 0) evecs(:, s0:e0) = block
            deallocate(block)
            deg = 0
         end if
      end do
      if (deg > 0) then
         s0 = m - deg
         e0 = m
         blk = e0 - s0 + 1
         allocate(block(m, blk)); block = evecs(:, s0:e0)
         call qr_orthonormal(m, blk, block, ierr)
         if (ierr == 0) evecs(:, s0:e0) = block
         deallocate(block)
      end if
   end subroutine apply_gs

   !> _calculate_alpha_matrix: alpha_ii = sqrt(G_ii / N_ii), N_ii = V_i^H G V_i.
   subroutine calc_alpha(m, evecs, gdiag, alpha)
      integer,          intent(in)  :: m
      complex(kind=cp), intent(in)  :: evecs(m, m)
      real(kind=dp),    intent(in)  :: gdiag(m)
      complex(kind=cp), intent(out) :: alpha(m)
      integer       :: i
      real(kind=dp) :: nii, gii, asq(m), zt2

      zt2 = ZERO_MATRIX_ELEMENT_THRESHOLD**2
      do i = 1, m
         nii = real(sum(conjg(evecs(:, i)) * gdiag * evecs(:, i)), dp)
         gii = gdiag(i)
         if (abs(nii) < zt2) then
            asq(i) = 0.0_dp
         else if (gii * nii < -zt2) then
            asq(i) = 0.0_dp
         else
            asq(i) = gii / nii
         end if
      end do
      where (asq < 0.0_dp) asq = 0.0_dp
      alpha = cmplx(sqrt(asq), 0.0_dp, kind=cp)
      where (abs(alpha) < ZERO_MATRIX_ELEMENT_THRESHOLD) alpha = (0.0_dp, 0.0_dp)
   end subroutine calc_alpha

   !> _match_and_reorder_minus_q: align -q modes to the +q basis by maximum
   !> projection, carry a phase factor onto the reordered -q alpha.
   subroutine match_reorder(n, Vp, alphap, Vm, evm, alpham, Vm_re, evm_re, alpham_re)
      integer,          intent(in)  :: n
      complex(kind=cp), intent(in)  :: Vp(2*n, 2*n), alphap(2*n)
      complex(kind=cp), intent(in)  :: Vm(2*n, 2*n), evm(2*n), alpham(2*n)
      complex(kind=cp), intent(out) :: Vm_re(2*n, 2*n), evm_re(2*n), alpham_re(2*n)

      integer          :: n2, ip, jm, best_j, ridx
      real(kind=dp)    :: nt2, ns2, proj, thr, tol, tol2
      complex(kind=cp) :: swapc(2*n, 2*n), target(2*n), source(2*n)
      complex(kind=cp) :: phase
      logical          :: matched(2*n)

      n2 = 2 * n
      tol = EIGENVECTOR_MATCHING_THRESHOLD
      tol2 = tol * tol
      Vm_re = (0.0_dp, 0.0_dp); evm_re = (0.0_dp, 0.0_dp); alpham_re = (0.0_dp, 0.0_dp)
      matched = .false.

      ! Vm_orig_swap_conj = conj( vstack( Vm[n:2n,:], Vm[0:n,:] ) )
      swapc(1:n, :)      = conjg(Vm(n+1:n2, :))
      swapc(n+1:n2, :)   = conjg(Vm(1:n, :))

      ! Loop 1: +q cols 1..n matched against swapc cols n+1..2n
      do ip = 1, n
         target = Vp(:, ip)
         nt2 = real(sum(conjg(target) * target), dp)
         if (nt2 < tol2) cycle
         best_j = 0; proj = -1.0_dp
         do jm = n + 1, n2
            if (matched(jm)) cycle
            source = swapc(:, jm)
            ns2 = real(sum(conjg(source) * source), dp)
            if (ns2 < tol2) cycle
            thr = sqrt(nt2 * ns2) - tol
            if (abs(sum(conjg(target) * source)) > thr) then
               if (abs(sum(conjg(target) * source)) > proj) then
                  proj = abs(sum(conjg(target) * source)); best_j = jm
               end if
            end if
         end do
         if (best_j > 0) then
            ridx = ip + n
            Vm_re(:, ridx) = Vm(:, best_j)
            evm_re(ridx)   = evm(best_j)
            call phase_factor(n2, target, swapc(:, best_j), tol, phase)
            alpham_re(ridx) = conjg(alphap(ip) * phase)
            matched(best_j) = .true.
         end if
      end do

      ! Loop 2: +q cols n+1..2n matched against swapc cols 1..n
      do ip = n + 1, n2
         target = Vp(:, ip)
         nt2 = real(sum(conjg(target) * target), dp)
         if (nt2 < tol2) cycle
         best_j = 0; proj = -1.0_dp
         do jm = 1, n
            if (matched(jm)) cycle
            source = swapc(:, jm)
            ns2 = real(sum(conjg(source) * source), dp)
            if (ns2 < tol2) cycle
            thr = sqrt(nt2 * ns2) - tol
            if (abs(sum(conjg(target) * source)) > thr) then
               if (abs(sum(conjg(target) * source)) > proj) then
                  proj = abs(sum(conjg(target) * source)); best_j = jm
               end if
            end if
         end do
         if (best_j > 0) then
            ridx = ip - n
            Vm_re(:, ridx) = Vm(:, best_j)
            evm_re(ridx)   = evm(best_j)
            call phase_factor(n2, target, swapc(:, best_j), tol, phase)
            alpham_re(ridx) = conjg(alphap(ip) * phase)
            matched(best_j) = .true.
         end if
      end do

      where (abs(alpham_re) < ZERO_MATRIX_ELEMENT_THRESHOLD) alpham_re = (0.0_dp, 0.0_dp)
   end subroutine match_reorder

   !> phase = (first |.|>tol component of target) / (first such of source).
   subroutine phase_factor(m, target, source, tol, phase)
      integer,          intent(in)  :: m
      complex(kind=cp), intent(in)  :: target(m), source(m)
      real(kind=dp),    intent(in)  :: tol
      complex(kind=cp), intent(out) :: phase
      integer          :: i
      complex(kind=cp) :: cvp0, cvm0
      logical          :: havep, havem

      havep = .false.; havem = .false.
      cvp0 = (0.0_dp, 0.0_dp); cvm0 = (0.0_dp, 0.0_dp)
      do i = 1, m
         if (.not. havep .and. abs(target(i)) > tol) then
            cvp0 = target(i); havep = .true.
         end if
         if (.not. havem .and. abs(source(i)) > tol) then
            cvm0 = source(i); havem = .true.
         end if
         if (havep .and. havem) exit
      end do
      if (havep .and. havem .and. abs(cvm0) > tol) then
         phase = cvp0 / cvm0
      else
         phase = (1.0_dp, 0.0_dp)
      end if
   end subroutine phase_factor

end module magcalc_kkd
